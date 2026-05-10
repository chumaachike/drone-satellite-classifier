# src/model.py
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 21, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None

    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def build_efficentnetb0(num_classes: int =21, pretrained: bool = True):
    weights =  models.EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(weights = weights)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model

def build_mobilenetsmall(num_classes: int=21, pretrained: bool = True):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT

    model = models.mobilenet_v3_small(weights= weights)

    model.classifier[3] = nn.Linear(
        model.classifer[3].in_features, num_classes
    )

    return model