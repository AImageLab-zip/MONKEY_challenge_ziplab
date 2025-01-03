import os
import time

from utils.data_preparation import DataPreparator
from utils.data_utils import get_device
from utils.logger import get_logger


class AbstractExperiment:
    def __init__(self, args, config):
        # -- PROJECT CONFIGs -- #
        self.args = args
        self.config = config
        self.logger = get_logger(name=self.__class__, args=args)
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

        self.output_dir = self.project_config.get("output_dir", "../outputs")

        # -- DATA CONFIGs -- #
        self.wsd_config = self.config.get("wholeslidedata", {})
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

        # -- TRAINING CONFIGs -- #
        self.training_config = self.config.get("training", {})
        if self.training_config is None:
            print("Training configurations not found in the configuration file.")
            return -1
        self.batch_size = self.wsd_config["default"].get("batch_size", 32)
        self.learning_rate = self.training_config.get("learning_rate", 0.001)
        self.epochs = self.training_config.get("epochs", 10)

        # -- CLASS STATE VARIABLES -- #
        self.data_prepator = None
        self.dataset_df = None
        self.folds_paths_dict = None
        self.model = None  # model object to store the model instance
        self.training_batch_generator = None

        # set-up the optional model params and gradient watch by wand-db (if enabled)
        # self.model_watch = getattr(self.args, "wandb_model_watch", False)

    def _prepare_data(self):
        self.data_prepator = DataPreparator(config=self.config)
        self.dataset_df, self.folds_paths_dict = self.data_prepator.prepare_data()

    def load_data(self):
        self._prepare_data()
        return self.dataset_df, self.folds_paths_dict

    def load_model(self):
        pass

    def save_model(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
