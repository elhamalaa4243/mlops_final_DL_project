
"""Model factory for supported pet-breed classification backbones."""

from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)

SUPPORTED_BACKBONES = (
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
)


def create_model(
    num_classes: int = 37,
    pretrained: bool = True,
    backbone: str = "resnet18",
):
    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )

    else:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            f"Choose from {SUPPORTED_BACKBONES}"
        )

    return model
