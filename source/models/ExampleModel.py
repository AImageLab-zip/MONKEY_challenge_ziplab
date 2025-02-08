import torch.nn as nn
from torchvision import models
from utils.logger import get_logger


class ExampleModel(nn.Module):
    def __init__(self, args, config):
        super(ExampleModel, self).__init__()
        self.args = args
        self.config = config
        # self.model_name = self.args.model
        # self.num_classes = getattr(self.args, "num_classes", 2)
        # self.freeze_all_layers = getattr(self.args, "freeze_all_layers", False)
        # self.pretrained = getattr(self.args, "pretrained", True)
        # self.weights = None
        # self.logger = get_logger(name=f"{self.model_name} model", args=self.args)

        # self.logger.info(f"Loading model with name: {self.model_name} ...")

        # self.model_names = {
        #     "resnet18": {
        #         "model": models.resnet18,
        #         "weights": models.ResNet18_Weights.DEFAULT,
        #     },
        #     "resnet34": {
        #         "model": models.resnet34,
        #         "weights": models.ResNet34_Weights.DEFAULT,
        #     },
        #     "resnet50": {
        #         "model": models.resnet50,
        #         "weights": models.ResNet50_Weights.DEFAULT,
        #     },
        #     "resnet101": {
        #         "model": models.resnet101,
        #         "weights": models.ResNet101_Weights.DEFAULT,
        #     },
        #     "resnet152": {
        #         "model": models.resnet152,
        #         "weights": models.ResNet152_Weights.DEFAULT,
        #     },
        # }

        # if self.model_name in self.model_names:
        #     if self.pretrained:
        #         self.logger.info(f"Loading {self.model_name} with pretrained weights!")
        #         self.weights = self.model_names[self.model_name]["weights"]
        #     else:
        #         self.logger.info(f"Loading {self.model_name} NON pretrained model!")

        #     self.model = self.model_names[self.model_name]["model"](
        #         weights=self.weights
        #     )

        # else:
        #     self.logger.error(
        #         f"Unsupported {self.model_name}network architecture specified!"
        #     )
        #     raise ValueError(f"Unsupported {self.model_name} specified!")

        # # Change the final layer for classification
        # self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        # # Freeze all layers
        # if self.freeze_all_layers:
        #     for param in self.model.parameters():
        #         param.requires_grad = False

        #     # unfreeze fc layers
        #     for param in self.model.fc.parameters():
        #         param.requires_grad = True

        # self.logger.info(f"{self.model_name} loaded successfully!")

    def forward(self, x):
        return self.model(x)

    def get_transforms(self):
        return self.weights.transforms() if self.weights else None
