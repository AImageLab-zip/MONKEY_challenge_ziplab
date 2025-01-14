import os

import detectron2.data.transforms as T
import torch
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from torchvision import models
from utils.logger import get_logger
from wholeslidedata.interoperability.detectron2.iterator import (
    WholeSlideDetectron2Iterator,
)
from wholeslidedata.interoperability.detectron2.predictor import (
    Detectron2DetectionPredictor,
)
from wholeslidedata.iterators import create_batch_iterator

# from wholeslidedata.interoperability.detectron2.trainer import (
#     WholeSlideDectectron2Trainer,
# ) #NOTE: added the import code of this object down here to make it customizable!


def transform(image):
    SIZE = 128
    AUG = T.FixedSizeCrop((SIZE, SIZE), pad_value=0)
    image = AUG.get_transform(image).apply_image(image)
    return image


class WholeSlideDectectron2Trainer(DefaultTrainer):
    def __init__(self, cfg, user_config, cpus):
        self._user_config = user_config
        self._cpus = cpus

        super().__init__(cfg)

    def build_train_loader(self, cfg):
        mode = "training"
        try:
            # print("User config passed to batch iterator:", self._user_config)
            training_batch_generator = create_batch_iterator(
                user_config=self._user_config,
                mode=mode,
                cpus=self._cpus,
                iterator_class=WholeSlideDetectron2Iterator,
            )
            assert training_batch_generator is not None, "Batch iterator returned None!"
            return training_batch_generator
        except Exception as e:
            print(f"Error while creating batch iterator: {e}")
            raise


class BatchPredictor(DefaultPredictor):
    """Run d2 on a list of images."""

    def __call__(self, images):
        input_images = []
        for image in images:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]
            new_image = transform(image)
            new_image = torch.as_tensor(new_image.astype("float32").transpose(2, 0, 1))

            input_images.append({"image": new_image, "height": height, "width": width})

        with torch.no_grad():
            preds = self.model(input_images)
        return preds


class Detectron2:
    def __init__(self, cfg, wsd_config, **kwargs):
        # super(Detectron2, self).__init__()

        self.cfg = cfg
        assert self.cfg, "Model configuration not found!"
        # extract the wsd_config from the kwargs
        self.wsd_config = wsd_config
        assert self.wsd_config, "WholeSlideData configuration not found!"

        self.INV_LABEL_MAP = {
            0: "lymphocyte",
            1: "monocyte",
        }

        self.num_workers = self.cfg.DATALOADER.get("NUM_WORKERS", 1)

        self.logger = get_logger(name="Detectron2")

        # initialize the model with the configuration
        self.model = build_model(self.cfg)

        # initialize the predictor for batch prediction
        self._predictor = BatchPredictor(self.cfg)

        # trainer
        self.trainer = None

    def load(self, resume=False):
        self.trainer = WholeSlideDectectron2Trainer(
            cfg=self.cfg, user_config=self.wsd_config, cpus=self.num_workers
        )
        self.trainer.resume_or_load(resume=resume)

    def train(self, resume=False):
        self.load(resume=resume)
        self.trainer.train()

    def predict_on_batch(self, x_batch):
        # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        outputs = self._predictor(x_batch)
        predictions = []
        for output in outputs:
            predictions.append([])
            pred_boxes = output["instances"].get("pred_boxes")

            scores = output["instances"].get("scores")
            classes = output["instances"].get("pred_classes")
            centers = pred_boxes.get_centers()
            for idx, center in enumerate(centers):
                x, y = center.cpu().detach().numpy()
                confidence = scores[idx].cpu().detach().numpy()
                label = self.INV_LABEL_MAP[int(classes[idx].cpu().detach())]
                prediction_record = {
                    "x": int(x),
                    "y": int(y),
                    "label": str(label),
                    "confidence": float(confidence),
                }
                predictions[-1].append(prediction_record)
        return predictions

        # other initializations, like the model

        # self.model = ...

    # def predict(self, data):
    #    pass

    # def forward(self, x):
    #     return self.model(x)

    # def get_transforms(self):
    #     return self.weights.transforms() if self.weights else None
