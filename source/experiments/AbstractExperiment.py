import os
import time
from copy import deepcopy

from tqdm import tqdm
from utils.data_preparation import DataPreparator
from utils.data_utils import get_device, load_yaml, px_to_mm, save_yaml, write_json_file
from utils.logger import get_logger
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import PatchConfiguration, create_patch_iterator


class AbstractExperiment:
    def __init__(self, args, config):
        # -- PROJECT CONFIGs -- #
        self.args = args
        self.config = config
        self.logger = get_logger(name=str(self.__class__.__name__), args=args)

        # debug flag
        self.debug = getattr(self.args, "debug", False)

        # get the device to use for computation
        self.device = get_device()

        # project configurations
        self.project_config = self.config.get("project", {})
        if self.project_config is None:
            print("Project configurations not found in the configuration file.")
            return -1  # TODO: implement better error handling

        self.seed = self.project_config.get("seed", 42)

        self.timestamp = getattr(
            self.args, "timestamp", time.strftime("%Y-%m-%d__%H-%M-%S")
        )
        # number of workers for the dataloaders
        self.num_workers = self.project_config.get("num_workers", "auto")
        if self.num_workers == "auto":
            self.num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))

        # -- DATA CONFIGs -- #
        self.dataset_configs = self.config.get("dataset", {})
        if self.dataset_configs is None:
            print("Dataset configurations not found in the configuration file.")
            return -1  # TODO: implement better error handling

        self.num_classes = self.dataset_configs.get("num_classes", 1)

        self.dataset_name = self.dataset_configs.get("name", "default_dataset")

        self.n_folds = self.dataset_configs.get("n_folds", 5)

        # -- MODEL CONFIGs -- #
        self.model_config = self.config.get("model", {})
        if self.model_config is None:
            print("Model configurations not found in the configuration file.")
            return -1  # TODO: implement better error handling

        self.model_name = self.model_config.get("name", None)
        self.pretrained = self.model_config.get("pretrained", False)

        self.conf_threshold = self.model_config.get("conf_threshold", 0.1)
        self.nms_threshold = self.model_config.get("nms_threshold", 0.3)

        # -- TRAINING CONFIGs -- #
        self.training_config = self.config.get("training", {})
        if self.training_config is None:
            print("Training configurations not found in the configuration file.")
            return -1

        self.batch_size = self.training_config.get("batch_size", 32)

        self.learning_rate = self.training_config.get("learning_rate", 0.001)
        self.epochs = self.training_config.get("epochs", 10)
        self.continue_training = getattr(self.args, "continue_training", False)

        self.model_dir = getattr(self.args, "model_dir", None)

        self.fold = getattr(self.args, "fold", None)
        if self.fold is not None:
            assert self.fold < self.n_folds, "Fold number should be less than n_folds."

        if self.continue_training and self.model_dir is None:
            if self.model_dir is None:
                raise ValueError(
                    "Model directory path must be provided if continue_training is True."
                )
        # -- OUTPUT CONFIGs -- #
        # pre-set up output directory for the experiment
        self.output_base_dir = self.project_config.get("output_dir", "../outputs")
        # self.experiment_name = f"{self.model_name}_e{self.epochs}_b{self.batch_size}_lr{self.learning_rate}_t{self.timestamp}"
        # self.output_dir = os.path.join(self.output_dir, self.experiment_name)
        self.experiment_name = None
        self.output_dir = None
        self.preds_dir = None
        self.patient_pred_dir = None

        # -- CLASS STATE CONSTANTS -- #
        self.SPACING_MIN = 0.25  # minimum rounded micro-meter per pixel spacing (resolution) of the whole slide images of the challenges
        self.SPACING_CONST = 0.24199951445730394  # maximum micro-meter per pixel spacing (resolution) of the whole slide images
        self.JSON_FILENAME_INFLAMMATORY_CELLS = "detected-inflammatory-cells.json"
        self.JSON_FILENAME_LYMPHOCYTES = "detected-lymphocytes.json"
        self.JSON_FILENAME_MONOCYTES = "detected-monocytes.json"

        # -- CLASS STATE VARIABLES -- #
        self.data_prepator = None
        self.dataset_df = None
        self.folds_paths_dict = None
        self.fold_yaml_paths_dict = None
        self.model = None  # model object to store the model instance
        self.training_batch_generator = None

        # set-up the optional model params and gradient watch by wand-db (if enabled)
        # self.model_watch = getattr(self.args, "wandb_model_watch", False)

        # # set up the wsd config dictionary, without wsi and wsa paths
        # self._set_wsd_config()

    def _set_wsd_config(self):
        self.wsd_config = deepcopy(self.config.get("wholeslidedata", {})["user_config"])

        if self.wsd_config is None:
            print(
                "Whole Slide Data configurations not found in the configuration file."
            )
            return -1  # TODO: implement better error handling

        # inject configs seed in the wsd_config
        self.wsd_config["wholeslidedata"]["default"]["seed"] = self.seed

        self.img_backend = self.wsd_config["wholeslidedata"]["default"].get(
            "image_backend", "openslide"
        )
        self.logger.info(f"Image backend: {self.img_backend}")

        # patches and shape configs
        # extract the patch shape and spacings from the wsd_config
        self.batch_shape = self.wsd_config["wholeslidedata"]["default"]["batch_shape"]
        # use the deep copy to avoid changing the original dict
        self.patch_shape = deepcopy(self.batch_shape.get("shape", (128, 128, 3)))
        self.spacings = deepcopy((self.batch_shape.get("spacing", 0.5),))
        self.y_shape = deepcopy(self.batch_shape.get("y_shape", (1000, 6)))

        # TODO: add the overlap and offset to the patch configuration yaml
        self.overlap = (0, 0)
        self.offset = (0, 0)
        self.center = False

        # inject batch size in the wsd config
        self.batch_shape["batch_size"] = self.batch_size

        # debug print for wsd config dict
        self.logger.debug(
            f"\n{10*'='}\nwsd_config debug: {self.wsd_config}\n{10*'='}\n"
        )

    def prepare_data(self):
        self.data_prepator = DataPreparator(config=self.config)
        self.dataset_df, self.folds_paths_dict = self.data_prepator.prepare_data()
        return self.dataset_df, self.folds_paths_dict

    def _load_fold(self, fold_path_dict):
        # set up the wsd config dictionary, without wsi and wsa paths
        self._set_wsd_config()

        self.fold_yaml_paths_dict = load_yaml(fold_path_dict)
        if self.fold_yaml_paths_dict is None:
            self.logger.error("Error loading fold yaml file.")
            return -1

        # inject fold splits (wsa and wsi paths) to the wsd config dict
        self.wsd_config["wholeslidedata"]["default"]["yaml_source"] = deepcopy(
            self.fold_yaml_paths_dict
        )

    def load_model(self):
        pass

    def save_model(self):
        pass

    def train(self):
        if self.dataset_df is None or self.folds_paths_dict is None:
            self.dataset_df, self.folds_paths_dict = self.prepare_data()

        # if fold is specified, train on that fold only
        if self.fold is not None:
            self.logger.info(
                "Training the model on single fold: {}...".format(self.fold)
            )
            self._load_fold(self.folds_paths_dict[self.fold])
            self.train_eval_fold(fold=self.fold)
            return 0
        # if no fold is specified, train on all folds
        self.logger.info("Training the model on all folds...")
        for fold, fold_path_dict in self.folds_paths_dict.items():
            self.logger.info(
                "Training fold {}/{}".format(fold + 1, len(self.folds_paths_dict))
            )

            # load the yaml file
            self._load_fold(fold_path_dict)

            self.train_eval_fold(fold=fold)
            self.logger.info(
                "Training of fold {}/{} completed".format(
                    fold + 1, len(self.folds_paths_dict)
                )
            )

        self.logger.info("Training of all folds completed")
        return 0

    def train_eval_fold(self, fold):
        pass

    def eval_fold(self, fold):
        self.logger.info(f"Evaluating the model on fold {fold}...")

        assert self.model is not None, "Model is not loaded."
        assert self.dataset_df is not None, "Dataset is not loaded."
        assert self.fold_yaml_paths_dict is not None, "Folds paths are not loaded."
        self.validation_fold_dict = self.fold_yaml_paths_dict["validation"]

        self.patch_configuration = PatchConfiguration(
            patch_shape=self.patch_shape,
            spacings=self.spacings,
            overlap=self.overlap,
            offset=self.offset,
            center=self.center,
        )

        progress_bar = tqdm(self.validation_fold_dict)

        for entry in progress_bar:
            wsi_path = entry["wsi"]["path"]
            # wsa_path = entry["wsa"]
            wsi_id = os.path.basename(wsi_path).split(".")[0]
            # remove the _PAS_CPG from the wsi_id
            if "_PAS_CPG" in wsi_id:
                wsi_id = wsi_id.split("_PAS_CPG")[0]

            mask_path = self.dataset_df.loc[
                self.dataset_df["Slide ID"] == wsi_id, "WSI Mask Path"
            ].values[0]
            progress_bar.set_description(f"Validating {wsi_id} ...")

            self.logger.debug(f"WSI path: {wsi_path}\nMask path: {mask_path}\n")

            immune_cells_dict, monocytes_dict, lymphocytes_dict = self.eval_wsi(
                wsi_path=str(wsi_path), mask_path=str(mask_path)
            )

            self._save_predictions(
                wsi_id=wsi_id,
                immune_cells_dict=immune_cells_dict,
                monocytes_dict=monocytes_dict,
                lymphocytes_dict=lymphocytes_dict,
            )

    def eval_wsi(self, wsi_path, mask_path):
        output_dict = {
            "name": "",
            "type": "Multiple points",
            "version": {"major": 1, "minor": 0},
            "points": [],
        }

        output_dict_immune_cells = deepcopy(output_dict)
        output_dict_lymphocytes = deepcopy(output_dict)
        output_dict_monocytes = deepcopy(output_dict)

        output_dict_immune_cells["name"] = "inflammatory-cells"
        output_dict_lymphocytes["name"] = "lymphocytes"
        output_dict_monocytes["name"] = "monocytes"

        annotations_immune_cells = []
        annotations_lymphocytes = []
        annotations_monocytes = []

        counter_immune_cells = 0
        counter_lymphocytes = 0
        counter_monocytes = 0

        iterator = create_patch_iterator(
            image_path=wsi_path,
            mask_path=mask_path,
            patch_configuration=self.patch_configuration,
            cpus=self.num_workers,
            backend="asap",  # needs to be ASAP for the path iterator to work in prediction
        )

        ratio = self.spacings[0] / self.SPACING_MIN
        with WholeSlideImage(wsi_path) as wsi:
            spacing = wsi.get_real_spacing(self.SPACING_MIN)
            print(
                f"Spacing: {spacing} - Spacing const: {self.SPACING_CONST} - ratio: {ratio}"
            )

        for x_batch, y_batch, info in tqdm(iterator):
            x_batch = x_batch.squeeze(0)
            y_batch = y_batch.squeeze(0)

            # predict points on the given batch
            predictions = self.model.predict_on_batch(x_batch)

            for idx, prediction in enumerate(predictions):
                c = info["x"]
                r = info["y"]

                for detections in prediction:
                    x, y, label, confidence = detections.values()
                    # print(f"Detected {label} at {x}, {y} with confidence {confidence}")

                    if x == 128 or y == 128:
                        continue

                    if y_batch[idx][y][x] == 0:
                        continue

                    x = (
                        x * ratio + c
                    )  # x is in spacing = 0.5 but c is in spacing = 0.25
                    y = y * ratio + r

                    prediction_record_immune_cells = {
                        "name": "Point " + str(counter_immune_cells),
                        "point": [
                            px_to_mm(x, spacing),
                            px_to_mm(y, spacing),
                            self.SPACING_CONST,
                        ],
                        "probability": confidence,
                    }
                    output_dict_immune_cells["points"].append(
                        prediction_record_immune_cells
                    )
                    annotations_immune_cells.append((x, y))
                    counter_immune_cells += 1

                    if label == "lymphocyte":  # lymphocyte
                        prediction_record_lymphocytes = {
                            "name": "Point " + str(counter_lymphocytes),
                            "point": [
                                px_to_mm(x, spacing),
                                px_to_mm(y, spacing),
                                self.SPACING_CONST,
                            ],
                            "probability": confidence,
                        }
                        output_dict_lymphocytes["points"].append(
                            prediction_record_lymphocytes
                        )
                        annotations_lymphocytes.append((x, y))
                        counter_lymphocytes += 1

                    elif label == "monocyte":  # monocyte
                        prediction_record_monocytes = {
                            "name": "Point " + str(counter_monocytes),
                            "point": [
                                px_to_mm(x, spacing),
                                px_to_mm(y, spacing),
                                self.SPACING_CONST,
                            ],
                            "probability": confidence,
                        }
                        output_dict_monocytes["points"].append(
                            prediction_record_monocytes
                        )
                        annotations_monocytes.append((x, y))
                        counter_monocytes += 1

                    else:
                        self.logger.warning("Unknown label")
                        continue

        # #TODO: bugged code, they had the same problem and even downgrading shapely didn't work :(
        # # saving xml file
        # annotations_wsd = to_wsd(annotations_immune_cells, label="inflammatory-cell")
        # xml_filename = 'points_results.xml'
        # output_path_xml = os.path.join(output_path,xml_filename)
        # write_point_set(
        #     annotations_wsd,
        #     output_path_xml,
        #     label_color="blue",
        # )

        return output_dict_immune_cells, output_dict_monocytes, output_dict_lymphocytes

    def _save_predictions(
        self,
        wsi_id,
        immune_cells_dict,
        monocytes_dict,
        lymphocytes_dict,
    ):
        # making output directories for saving the predictions
        self.preds_dir = os.path.join(self.output_dir, "results")
        self.patient_pred_dir = os.path.join(self.preds_dir, wsi_id)

        # create the patient prediction directory if it doesn't exist
        os.makedirs(self.patient_pred_dir, exist_ok=True)

        # saving the json files for immune, monocytes and lymphocytes predictions
        write_json_file(
            json_dict=immune_cells_dict,
            save_dir=self.patient_pred_dir,
            file_name=self.JSON_FILENAME_INFLAMMATORY_CELLS,
        )

        write_json_file(
            json_dict=monocytes_dict,
            save_dir=self.patient_pred_dir,
            file_name=self.JSON_FILENAME_MONOCYTES,
        )

        write_json_file(
            json_dict=lymphocytes_dict,
            save_dir=self.patient_pred_dir,
            file_name=self.JSON_FILENAME_LYMPHOCYTES,
        )

    def test(self):
        pass
