import torch
import torch.nn as nn
from torchvision.models import resnet18

device = 'cuda:0'
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18FeatureExtractor, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
    def forward(self, x):
        features = self.resnet(x)
        features = features.to(device)
        return features