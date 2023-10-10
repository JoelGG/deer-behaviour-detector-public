import torchvision.models as models
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import torch.nn as nn

resnets = {
    "r2plus1d_18": models.video.r2plus1d_18,
    "r3d_18": models.video.r3d_18,
    "mc3_18": models.video.mc3_18,
}


class Identity(nn.Module):
    """
    Representation of the identity operation as a PyTorch `nn.Module` class.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fine_tune_resnet(feature_extract: bool, pretrained=False, resnet_type="r3d_18"):
    """
    Parameters
    __________
    feature_extract : bool
        If true then all model weights are frozen
    pretrained :
        If true then instantiate model with default pretrained weights
    """
    model = resnets[resnet_type](weights="DEFAULT" if pretrained else None)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.avgpool = Identity()
    model.fc = Identity()
    return model, num_ftrs


def fine_tune_mvit(feature_extract, pretrained=False):
    """
    Parameters
    __________
    feature_extract : bool
        If true then all model weights are frozen
    pretrained :
        If true then instantiate model with default pretrained weights
    """
    model = mvit_v2_s(weights=MViT_V2_S_Weights.DEFAULT if pretrained else None)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.head[1].in_features
    model.head = Identity()
    return model, num_ftrs


def set_parameter_requires_grad(model, feature_extracting):
    """
    Parameters
    __________
    model : nn.Module
        Module which will have its weights adjusted
    feature_extracting :
        If true then all model weights are frozen
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
