####Create ResNet-18

from torch import nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


def create_model(num_classes=37, pretrained=True):
    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained else None
    )

    model = resnet18(weights=weights)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)

    return model