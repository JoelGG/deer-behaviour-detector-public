import torchvision.models as models
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import torch
import torch.nn as nn

resnets = {
    "r2plus1d_18": models.video.r2plus1d_18,
    "r3d_18": models.video.r3d_18,
    "mc3_18": models.video.mc3_18,
}


class DeerTripleStream(nn.Module):
    def __init__(self, num_classes=2, feature_extract=False, pretrained=False):
        super(DeerTripleStream, self).__init__()

        hidden_size = 100
        self.mvit_rgb, mvit_rgb_in_features = fine_tune_mvit(
            num_classes, feature_extract, pretrained
        )
        self.mvit_rgb_cropped, mvit_rgb_cropped_in_features = fine_tune_mvit(
            num_classes, feature_extract, pretrained
        )
        self.r18_flow_cropped, r18_flow_cropped_in_features = fine_tune_resnet(
            feature_extract, False
        )
        self.fc1 = nn.Linear(
            in_features=(
                mvit_rgb_in_features
                + mvit_rgb_cropped_in_features
                + r18_flow_cropped_in_features
            ),
            out_features=hidden_size,
            bias=True,
        )
        self.dropout = nn.Dropout(p=0)
        self.fc2 = nn.Linear(
            in_features=hidden_size,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, rgb_frames, rgb_frames_cropped, flow_frames_cropped):
        x_rgb = self.mvit_rgb(rgb_frames)
        x_rgb_cropped = self.mvit_rgb_cropped(rgb_frames_cropped)
        x_flow_cropped = self.r18_flow_cropped(rgb_frames_cropped)

        x = torch.cat((x_rgb, x_rgb_cropped, x_flow_cropped), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeerSpatial(nn.Module):
    def __init__(self, num_classes=2, feature_extract=False, pretrained=False):
        super(DeerSpatial, self).__init__()
        self.resnet, in_features = fine_tune_resnet(feature_extract, pretrained)
        self.dropout = nn.Dropout(p=0)
        self.fc = nn.Linear(
            in_features=in_features, out_features=num_classes, bias=True
        )

    def forward(self, x, _):
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DeerSpatialMViT(nn.Module):
    def __init__(self, num_classes=2, feature_extract=False, pretrained=False):
        super(DeerSpatialMViT, self).__init__()
        self.upsample = nn.Upsample(size=(16, 224, 224))
        self.mvit, in_features = fine_tune_mvit(feature_extract, pretrained)
        self.dropout = nn.Dropout(p=0)
        self.fc = nn.Linear(
            in_features=in_features, out_features=num_classes, bias=True
        )

    def forward(self, x, _):
        x = self.upsample(x)
        x = self.mvit(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DeerTwoStreamMViT(nn.Module):
    def __init__(
        self, num_classes=2, feature_extract=False, pretrained=False, hidden_size=100
    ):
        super(DeerTwoStreamMViT, self).__init__()
        self.mvit, in_features_mvit = fine_tune_mvit(feature_extract, pretrained)
        self.resnet, in_features_resnet = fine_tune_resnet(False, False)
        self.fc1 = nn.Linear(
            in_features=in_features_mvit + in_features_resnet,
            out_features=hidden_size,
            bias=True,
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(
            in_features=hidden_size,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, x_rgb, x_flow):
        spacial_ratio = 8
        clip_len = x_rgb.shape[-3]
        spatial_clip_len = clip_len // spacial_ratio

        x_rgb = self.mvit(x_rgb)
        x_flow = self.resnet(subsample(x_flow, spatial_clip_len))
        x = torch.cat((x_rgb, x_flow), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeerTwoStream(nn.Module):
    def __init__(
        self, num_classes=2, spacial_ratio=8, feature_extract=False, pretrained=False
    ):
        super(DeerTwoStream, self).__init__()
        self.spacial_ratio = spacial_ratio

        self.resnet_spacial, spacial_in_features = fine_tune_resnet(
            feature_extract, pretrained
        )
        self.resnet_temporal, temporal_in_features = fine_tune_resnet(
            feature_extract, False
        )
        self.fc = nn.Linear(
            in_features=spacial_in_features + temporal_in_features,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, rgb_input, flow_input):
        clip_len = rgb_input.shape[-3]
        spatial_clip_len = clip_len // self.spacial_ratio

        x_temporal = subsample(flow_input, spatial_clip_len)
        x_spacial = rgb_input

        x_temporal = self.resnet_temporal(x_temporal)
        x_spacial = self.resnet_spacial(x_spacial)
        x = torch.cat((x_spacial, x_temporal), dim=1)
        x = self.fc(x)
        return x


class DeerTwoStreamFlow(nn.Module):
    def __init__(self, num_classes=2, feature_extract=False, pretrained=False):
        super(DeerSpatial, self).__init__()

        self.resnet_spacial, spacial_in_features = fine_tune_resnet(
            feature_extract, pretrained
        )
        self.resnet_temporal, temporal_in_features = fine_tune_resnet(
            feature_extract, pretrained
        )
        self.fc = nn.Linear(
            in_features=spacial_in_features + temporal_in_features,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, rgb_input, flow_input):
        x_spacial = self.resnet_spacial(x)
        x_temporal = self.resnet_temporal(flow_input)
        x = torch.cat((x_spacial, x_temporal), dim=1)
        x = self.fc(x)
        return x


class Identity(nn.Module):
    """
    Representation of the identity operation as a PyTorch `nn.Module` class.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fine_tune_resnet(feature_extract, pretrained=False):
    """
    Parameters
    __________
    feature_extract : bool
        If true then all model weights are frozen
    pretrained :
        If true then instantiate model with default pretrained weights
    """
    model = models.video.r3d_18(weights="DEFAULT" if pretrained else None)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
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


def subsample(x, num_samples, temporal_dim=-3):
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    indices = indices.to(x.device)
    output = torch.index_select(x, temporal_dim, indices)
    return output
