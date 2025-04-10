import torch
import torch.amp
import torch.nn as nn
import torchvision
import pdb
import os

from instanseg.utils.utils import show_images, display_as_grid, download_model
from instanseg.utils.pytorch_utils import torch_fastremap
from pytorch_utils import get_masked_patches
from instanseg.utils.biological_utils import get_mean_object_features
import torch
import tifffile
import matplotlib.pyplot as plt

class PatchClassifier(nn.Module):
    def __init__(self, num_classes = 16,
                 embedding_dim = 64,
                 patch_size = 128,
                 pretrained=False,
                 encoder = 'mobilenet_v3',
                 from_file = None,
                 dropprob = 0.1,
                 dim_in = 4,
                 ):
        super(PatchClassifier, self).__init__()
 
        if isinstance(num_classes,list):
            num_classes = sum(num_classes)
 
        if encoder.startswith("resnet18"):
            if encoder.startswith("resnet18_original"):
                self.model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)
                self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)
                self.model.conv1 = torch.nn.Conv2d(dim_in,64,(7,7),  padding=(3, 3), bias=False)
                self.classifier = nn.Linear(embedding_dim, num_classes)
            else:
                self.model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)
                self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)
                self.model.conv1 = torch.nn.Conv2d(dim_in,64,(7,7), stride = (1,1), padding=(3, 3), bias=False)
                self.classifier = nn.Linear(embedding_dim, num_classes)
 
        elif encoder.startswith("efficientnet"):
 
            if encoder.endswith("l"):
             #   import torchvision
                self.model = torchvision.models.efficientnet_v2_l('IMAGENET1K_V1')
                self.model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
 
            else:
              #  import torchvision
                self.model = torchvision.models.efficientnet_v2_m('IMAGENET1K_V1')
                self.model.features[0][0] = nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
 
            self.model.classifier = nn.Identity()
            in_feat = 1280
                    
        elif encoder.startswith("mobilenetv4"):
            import timm
 
            if "hybrid" in encoder:
                self.model = timm.create_model('mobilenetv4_hybrid_medium')
                self.model.conv_stem = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), bias=False)
            
            else:
                self.model = timm.create_model('mobilenetv4_conv_large')
                self.model.conv_stem = torch.nn.Conv2d(4, 24, kernel_size=(3, 3), stride=(1, 1), bias = False) #default is onv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.model.classifier = nn.Identity()
            # self.classifier = nn.Linear(embedding_dim, num_classes)
            in_feat = 1280
 
        elif encoder in ['mobilenet_v3','mobilenet_v3_large','mobilenet_v3_original','mobilenet_v3_large_original']:
            if encoder.startswith('mobilenet_v3_large'):
                self.model = torchvision.models.mobilenet_v3_large(norm_layer = nn.BatchNorm2d )
            else:
                self.model = torchvision.models.mobilenet_v3_small(norm_layer = nn.BatchNorm2d )
 
            if not encoder.endswith('original'):
                self.model.features[0][0] = torch.nn.Conv2d(dim_in, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            else:
                self.model.features[0][0] = torch.nn.Conv2d(dim_in, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
 
            in_feat = self.model.classifier[0].in_features
            self.model.classifier = nn.Identity()
 
        self.classifier = nn.Sequential(
            nn.Linear(in_feat, embedding_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropprob, inplace=True),
            nn.Linear(embedding_dim, num_classes),
        )
 
        if from_file is not None:
            
            file_path = from_file
            checkpoint = torch.load(file_path)
            self.load_state_dict(checkpoint['model_state_dict'], strict = False)
            print(f"Loaded model from {from_file}")

        self.patch_size = patch_size
        self.num_classes = num_classes


    #@torch.jit.unused
    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x
    
    
    def embed(self, x):

        with torch.no_grad():
            x = self.model(x)
            x = self.classifier[0](x)
        return x
    

