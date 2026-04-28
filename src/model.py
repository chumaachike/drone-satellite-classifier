# src/model.py
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 21, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None

    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model