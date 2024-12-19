import torch.nn as nn
from torchvision import models
from utils.logger import get_logger


class Detectron2(nn.Module):
    def __init__(self, args, config):
        super(Detectron2, self).__init__()
        self.args = args
        self.config = config
        self.model_conf = self.config.get("model", None)
        assert self.model_conf is not None, "Model configuration not found!"

        #other initializations, like the model
        
        # self.model = ...
            
    # def predict(self, data):
    #    pass
    
    # def forward(self, x):
    #     return self.model(x)

    # def get_transforms(self):
    #     return self.weights.transforms() if self.weights else None
