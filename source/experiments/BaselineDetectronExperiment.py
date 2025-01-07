import os
import pprint

import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from utils.data_utils import load_yaml, save_yaml
from wholeslidedata.interoperability.detectron2.iterator import (
    WholeSlideDetectron2Iterator,
)
from wholeslidedata.interoperability.detectron2.predictor import (
    Detectron2DetectionPredictor,
)
from wholeslidedata.interoperability.detectron2.trainer import (
    WholeSlideDectectron2Trainer,
)
from wholeslidedata.iterators import create_batch_iterator

# from wholeslidedata.visualization.plotting import plot_boxes
from .AbstractExperiment import AbstractExperiment


class BaselineDetectronExperiment(AbstractExperiment):
    def __init__(self, args, config):
        super().__init__(args, config)  # Call the constructor of the parent class

        self.config_url = self.model_config.get("config_url", None)
        if self.config_url is None:
            self.config_url = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            self.logger.warning(
                "Using default config file for the model -> {}".format(self.config_url)
            )
        self.cfg = get_cfg()
        # using faster rcnn architecture
        self.cfg.merge_from_file(model_zoo.get_config_file(self.config_url))
        if self.model_config.get("pretrained", False):
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                self.config_url
            )  # download the model weights to fine-tune

        self.cfg.SEED = self.seed  # set the seed for reproducibility

        # self.cfg.DATASETS.TRAIN = (self.dataset_name + "_train",)
        # self.cfg.DATASETS.TEST = (self.dataset_name + "_val",)
        self.cfg.DATASETS.TRAIN = ("detection_dataset2",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = self.num_workers

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            512  # was 512 #TODO: is this correct?
        )
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes  # was 1

        self.cfg.SOLVER.IMS_PER_BATCH = (
            self.batch_size
        )  # was 10 #TODO: is this correct?
        self.cfg.SOLVER.BASE_LR = self.learning_rate  # pick a good, was 0.001
        self.cfg.SOLVER.MAX_ITER = self.epochs  # 2000 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset

        self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
            [16, 24, 32]
        ]  # TODO: don't hardcode this
        self.cfg.SOLVER.STEPS = (10, 100, 250)  # TODO: don't hardcode this
        self.cfg.SOLVER.WARMUP_ITERS = 0  # TODO: don't hardcode this
        self.cfg.SOLVER.GAMMA = 0.5  # TODO: don't hardcode this

        self.cfg.OUTPUT_DIR = self.output_dir
        # create the output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # save the config file
        save_yaml(self.cfg, save_dir=self.output_dir, file_name="config.yaml")

    def train_fold(self, fold, fold_path_dict):
        # check if model directory is provided else use the output directory
        if self.model_dir is not None:
            self.output_dir = self.model_dir

        self.model_path = os.path.join(self.output_dir, f"fold_{fold}")
        # make the directory for the fold
        os.makedirs(self.model_path, exist_ok=True)

        # TODO: bug here? if we use a pretrained model, we should not overwrite the weights?
        # set the model weights path
        # self.cfg.MODEL.WEIGHTS = os.path.join(
        #     self.model_path, f"model_final_{fold}.pth"
        # )

        # check if model already exists
        if os.path.exists(self.cfg.MODEL.WEIGHTS):
            self.logger.info(f"Model for fold {fold} already exists.")
            if not self.continue_training:
                self.logger.info("Skipping fold training.")
                return
            else:
                self.logger.info("Continuing training from existing model.")

        # load the yaml file
        self.fold_yaml_paths_dict = load_yaml(fold_path_dict)
        if self.fold_yaml_paths_dict is None:
            self.logger.error("Error loading fold yaml file.")
            return -1

        # self.fold_training_yaml_paths_dict = self.fold_yaml_paths_dict["training"]
        # self.fold_validation_yaml_paths_dict = self.fold_yaml_paths_dict["validation"]

        # inject fold splits to the config dict
        self.wsd_config["wholeslidedata"]["default"]["yaml_source"] = (
            self.fold_yaml_paths_dict
        )

        # save the updated config to the model directory
        save_yaml(
            self.wsd_config,
            save_dir=self.model_path,
            file_name=f"wsd_config_fold_{fold}.yaml",
        )

        # self.wsd_config["wholeslidedata"]["default"]["yaml_source"] = (
        #     self.fold_training_yaml_paths_dict
        # )

        # self.logger.debug(self.wsd_config["wholeslidedata"]["train"])

        self.model = build_model(self.cfg)

        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info("Parameter Count:\n" + str(pytorch_total_params))

        self.trainer = WholeSlideDectectron2Trainer(
            cfg=self.cfg, user_config=self.wsd_config, cpus=self.num_workers
        )
        self.trainer.resume_or_load(resume=self.continue_training)
        self.trainer.train()

    def _predict(self):
        pass

    def test(self):
        pass
