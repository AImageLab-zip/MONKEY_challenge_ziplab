import os
import time
from copy import deepcopy

from tqdm import tqdm
from utils.data_preparation import DataPreparator
from utils.data_utils import get_device, px_to_mm, write_json_file
from utils.logger import get_logger
from wholeslidedata.image.wholeslideimage import WholeSlideImage


class AbstractExperiment:
    def __init__(self, args, config):
        # -- PROJECT CONFIGs -- #
        self.args = args
        self.config = config
        self.logger = get_logger(name=str(self.__class__), args=args)

        # debug flag
        self.debug = getattr(self.args, "debug", False)

        # get the device to use for computation
        self.device = get_device()

        # project configurations
        self.project_config = self.config.get("project", {})
        if self.project_config is None:
            print("Project configurations not found in the configuration file.")
            return -1  # TODO: implement better error handling
        # seed
        self.seed = self.project_config.get("seed", 42)

        self.timestamp = getattr(
            self.args, "timestamp", time.strftime("%Y-%m-%d__%H-%M-%S")
        )
        # number of workers for the dataloaders
        self.num_workers = self.project_config.get("num_workers", "auto")
        if self.num_workers == "auto":
            self.num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))

        # -- DATA CONFIGs -- #
        self.wsd_config = self.config.get("wholeslidedata", {})["user_config"]
        self.logger.debug(f"wsd_config: {self.wsd_config}")
        if self.wsd_config is None:
            print(
                "Whole Slide Data configurations not found in the configuration file."
            )
            return -1  # TODO: implement better error handling

        self.dataset_configs = self.config.get("dataset", {})
        if self.dataset_configs is None:
            print("Dataset configurations not found in the configuration file.")
            return -1  # TODO: implement better error handling

        self.num_classes = self.dataset_configs.get("num_classes", 1)

        self.dataset_name = self.dataset_configs.get("name", "default_dataset")

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
        # inject batch size in the wsd config
        self.wsd_config["wholeslidedata"]["default"]["batch_shape"]["batch_size"] = (
            self.batch_size
        )

        self.learning_rate = self.training_config.get("learning_rate", 0.001)
        self.epochs = self.training_config.get("epochs", 10)
        self.continue_training = getattr(self.args, "continue_training", False)

        self.model_dir = getattr(self.args, "model_dir", None)

        if self.continue_training and self.model_dir is None:
            if self.model_dir is None:
                raise ValueError(
                    "Model directory path must be provided if continue_training is True."
                )
        # -- OUTPUT CONFIGs -- #
        # set up unique output directory for the experiment
        self.output_dir = self.project_config.get("output_dir", "../outputs")
        # self.experiment_name = f"{self.model_name}_e{self.epochs}_b{self.batch_size}_lr{self.learning_rate}_t{self.timestamp}"
        # self.output_dir = os.path.join(self.output_dir, self.experiment_name)

        # -- CLASS STATE VARIABLES -- #
        self.data_prepator = None
        self.dataset_df = None
        self.folds_paths_dict = None
        self.fold_yaml_paths_dict = None
        self.model = None  # model object to store the model instance
        self.training_batch_generator = None

        # set-up the optional model params and gradient watch by wand-db (if enabled)
        # self.model_watch = getattr(self.args, "wandb_model_watch", False)

    def load_data(self):
        self.data_prepator = DataPreparator(config=self.config)
        self.dataset_df, self.folds_paths_dict = self.data_prepator.prepare_data()
        return self.dataset_df, self.folds_paths_dict

    def load_model(self):
        pass

    def save_model(self):
        pass

    def train(self):
        self.dataset_df, self.folds_paths_dict = self.load_data()

        self.logger.info("Training the model...")
        for fold, fold_path_dict in self.folds_paths_dict.items():
            self.logger.info(
                "Training fold {}/{}".format(fold + 1, len(self.folds_paths_dict))
            )
            self.train_eval_fold(fold=fold, fold_path_dict=fold_path_dict)
            self.logger.info(
                "Training of fold {}/{} completed".format(
                    fold + 1, len(self.folds_paths_dict)
                )
            )

    def train_eval_fold(self, fold, fold_path_dict):
        pass

    def eval_fold(self, fold, fold_path_dict):
        self.logger.info(f"Evaluating the model on fold {fold}...")

        self.validation_fold_dict = fold_path_dict["validation"]

        progress_bar = tqdm(self.validation_fold_dict)

        for wsi_path, wsa_path in progress_bar:
            wsi_id = os.path.basename(wsi_path).split(".")[0]
            progress_bar.set_description(f"Validating {wsi_id} ...")

            # TODO: don't hardcode those values
            patch_shape = (128, 128, 3)
            spacings = (0.5,)
            overlap = (0, 0)
            offset = (0, 0)
            center = False

            self.patch_configuration = PatchConfiguration(
                patch_shape=patch_shape,
                spacings=spacings,
                overlap=overlap,
                offset=offset,
                center=center,
            )

            iterator = create_patch_iterator(
                image_path=wsi_path,
                mask_path=wsa_path,
                patch_configuration=self.patch_configuration,
                cpus=self.num_workers,
                backend="openslide",
            )  # was backend='asap'
            immune_cells_dict, monocytes_dict, lymphocytes_dict = self.eval_wsi(iterator=iterator,)

    def eval_wsi(self, iterator, predictor, spacing, image_path, output_path):
        SPACING_CONST = 0.24199951445730394

        json_filename_immune_cells = "detected-inflammatory-cells.json"
        json_filename_lymphocytes = "detected-lymphocytes.json"
        json_filename_monocytes = "detected-monocytes.json"

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
        spacing_min = 0.25  # was used in the original code to edit the annotations to bounding boxes

        ratio = spacing / spacing_min
        with WholeSlideImage(image_path) as wsi:
            spacing = wsi.get_real_spacing(spacing_min)
            print(
                f"Spacing: {spacing} - Spacing const: {SPACING_CONST} - ratio: {ratio}"
            )

        for x_batch, y_batch, info in tqdm(iterator):
            x_batch = x_batch.squeeze(0)
            y_batch = y_batch.squeeze(0)

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
                            SPACING_CONST,
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
                                SPACING_CONST,
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
                                SPACING_CONST,
                            ],
                            "probability": confidence,
                        }
                        output_dict_monocytes["points"].append(
                            prediction_record_monocytes
                        )
                        annotations_monocytes.append((x, y))
                        counter_monocytes += 1

                    else:
                        print("Unknown label")
                        continue

        print(f"Predicted {len(annotations_immune_cells)} points")
        print("saving predictions...")

        # for i, points in enumerate(annotations):
        #     print(f"Annotation {i}: {points}")

        # saving json file immune cells
        output_path_json_immune_cells = os.path.join(
            output_path, json_filename_immune_cells
        )
        write_json_file(
            location=output_path_json_immune_cells, content=output_dict_immune_cells
        )

        # saving json file lymphocytes
        output_path_json_lyphocytes = os.path.join(
            output_path, json_filename_lymphocytes
        )
        write_json_file(
            location=output_path_json_lyphocytes, content=output_dict_lymphocytes
        )

        # saving json file monocytes
        output_path_json_monocytes = os.path.join(output_path, json_filename_monocytes)
        write_json_file(
            location=output_path_json_monocytes, content=output_dict_monocytes
        )

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

        print("finished!")

    def test(self):
        pass
