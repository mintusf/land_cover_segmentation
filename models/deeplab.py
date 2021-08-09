""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn


def create_deeplab(channels_in: int = 3, channels_out=13):
    """DeepLabv3 class with custom head
    Args:
        channels_in: number of input channels
        channels_out: number of output channels
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=False)

    if channels_in != 3:
        model.backbone.conv1 = nn.Conv2d(
            channels_in, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    model.classifier = DeepLabHead(2048, channels_out)

    # Set the model in training mode
    model.train()

    return model
