import os
import pprint

import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from models import ModelFactory
from utils.data_utils import load_yaml, save_yaml

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
        if self.pretrained:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                self.config_url
            )  # download the model weights to fine-tune

        self.cfg.SEED = self.seed  # set the seed for reproducibility
        self.cfg.DATASETS.TRAIN = (self.dataset_name + "_train",)
        # self.cfg.DATASETS.TEST = (self.dataset_name + "_val",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = self.num_workers
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_threshold
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.nms_threshold
        self.cfg.MODEL.RPN.NMS_THRESH = self.nms_threshold

        self.cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        self.cfg.SOLVER.BASE_LR = self.learning_rate
        self.cfg.SOLVER.MAX_ITER = self.epochs  # 2000 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset

        # NOTE: hardcoding the values for now!
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # was 512
        self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24, 32]]
        # self.cfg.SOLVER.STEPS = (10, 100, 250)
        self.cfg.SOLVER.WARMUP_ITERS = 0
        self.cfg.SOLVER.GAMMA = 0.5

        # - OUTPUTs CONFIGs -#

        # check if model directory is provided else use the output directory
        if self.model_dir is not None:
            self.output_dir = self.model_dir
        else:
            self.experiment_name = f"{self.model_name}_pretrained_{self.pretrained}_e{self.cfg.SOLVER.MAX_ITER}_b{self.cfg.SOLVER.IMS_PER_BATCH}_lr{self.cfg.SOLVER.BASE_LR}"
            self.output_dir = os.path.join(self.output_base_dir, self.experiment_name)

        # create the output directory
        os.makedirs(self.output_dir, exist_ok=True)
        # save the config file
        save_yaml(self.cfg, save_dir=self.output_dir, file_name="model_config.yaml")

    def train_eval_fold(self, fold):
        # set the fold path
        self.fold_path = os.path.join(self.output_dir, f"fold_{fold}")
        # make the directory for the fold
        os.makedirs(self.fold_path, exist_ok=True)
        # set the output directory for the fold for detectron2
        self.cfg.OUTPUT_DIR = self.fold_path

        # TODO: bug here? if we use a pretrained model, we should not overwrite the weights?
        # set the model weights path
        # self.cfg.MODEL.WEIGHTS = os.path.join(
        #     self.fold_path, f"model_final_{fold}.pth"
        # )

        # check if model already exists, and decide if continuing training
        if os.path.exists(self.cfg.MODEL.WEIGHTS):
            self.logger.info(f"Model for fold {fold} already exists.")
            if not self.continue_training:
                self.logger.info("Skipping fold training.")
                return
            else:
                self.logger.info("Continuing training from existing model.")

        # save the updated config to the model directory
        save_yaml(
            self.wsd_config,
            save_dir=self.fold_path,
            file_name=f"wsd_config_fold_{fold}.yaml",
        )

        # create the model instance
        self.model = ModelFactory().get_model(
            self.model_name, cfg=self.cfg, wsd_config=self.wsd_config
        )

        # train the model on the fold and continue the training if required
        self.model.train(resume=self.continue_training)

        # evaluate the model on the evaluation set for the selected fold
        self.eval_fold(fold=fold)

    def _predict(self):
        pass

    def test(self):
        pass
