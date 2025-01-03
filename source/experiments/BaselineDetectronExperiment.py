from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
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
        
        self.config_url = self.model_config.get(
            "config_url", None
        )
        if self.config_url is None:
            self.config_url = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            self.logger.warning("Using default config file for the model -> {}".format(self.config_url))        
        self.cfg = get_cfg()
        # using faster rcnn architecture
        self.cfg.merge_from_file(
            model_zoo.get_config_file(self.config_url)
        )
        if self.model_config.get("pretrained", False):
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                self.config_url
            )  # download the model weights to fine-tune
        
        self.cfg.DATASETS.TRAIN = (self.dataset_name + "_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = self.num_workers

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # was 512
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self. # was 1
        self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24, 32]]

        self.cfg.SOLVER.IMS_PER_BATCH = 10
        self.cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
        self.cfg.SOLVER.MAX_ITER = 2000  # 2000 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        self.cfg.SOLVER.STEPS = (10, 100, 250)
        self.cfg.SOLVER.WARMUP_ITERS = 0
        self.cfg.SOLVER.GAMMA = 0.5

        self.cfg.OUTPUT_DIR = str(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # save the config file
        cfg_file = output_folder / "config.yaml"
        with open(cfg_file, "w") as f:
            f.write(cfg.dump())

    def train(self):
        self.training_batch_generator = create_batch_iterator(
            user_config=self.wsd_config,
            mode="training",
            cpus=self.num_workers,
            iterator_class=WholeSlideDetectron2Iterator,
        )

    def _predict(self):
        pass

    def test(self):
        pass
