import os
import time
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from wholeslidedata.interoperability.detectron2.iterator import WholeSlideDetectron2Iterator
from wholeslidedata.interoperability.detectron2.trainer import WholeSlideDectectron2Trainer
from wholeslidedata.interoperability.detectron2.predictor import Detectron2DetectionPredictor
from wholeslidedata.iterators import create_batch_iterator
from wholeslidedata.visualization.plotting import plot_boxes

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model

def main():


    cwd = os.getcwd()
    print('current directory is:', cwd)

    #########################################################################################
    # defining training configuration\
    # it can also be defined in an extra yaml file

    user_config = {
        'wholeslidedata': {
            'default': {
                'yaml_source': "./configs/training_all_infcell.yml",
                "seed": 42,
                "image_backend": "asap",
                'labels': {
                    "ROI": 0,
                    "inf_cell": 1
                },

                'batch_shape': {
                    'batch_size': 10,
                    'spacing': 0.5,
                    'shape': [128, 128, 3],
                    'y_shape': [1000, 6],
                },

                "annotation_parser": {
                    "sample_label_names": ['roi'],
                },

                'point_sampler_name': "RandomPointSampler",
                'point_sampler': {
                    "buffer": {'spacing': "${batch_shape.spacing}", 'value': -64},
                },

                'patch_label_sampler_name': 'DetectionPatchLabelSampler',
                'patch_label_sampler': {
                    "max_number_objects": 1000,
                    "detection_labels": ['inf_cell'],

                },

            }
        }
    }

    #########################################################################################
    # creating output folder for saving the model and results:

    output_folder = Path('./outputs')
    if not (os.path.isdir(output_folder)): os.mkdir(output_folder)
    cpus = 4

    #########################################################################################
    # Train the model:

    cfg = get_cfg()
    # using faster rcnn architecture
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )

    cfg.DATASETS.TRAIN = ("detection_dataset2",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 24, 32]]

    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 200000
    cfg.SOLVER.STEPS = (10000, 25000, 50000 ,100000, 150000) #(10, 100, 250)
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.GAMMA = 0.5

    # Set checkpoint saving interval
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000

    cfg.OUTPUT_DIR = str(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameter Count:\n" + str(pytorch_total_params))

    trainer = WholeSlideDectectron2Trainer(cfg, user_config=user_config, cpus=cpus)
    trainer.resume_or_load(resume=False) #resume=False
    trainer.train()

    #########################################################################################


if __name__ == "__main__":
    main()