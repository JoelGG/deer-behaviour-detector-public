import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from deerbehaviourdetector.models.backbones import fine_tune_mvit

from models.backbones import fine_tune_resnet


class DeerSingleStreamResnet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        feature_extract=False,
        pretrained=False,
        resnet_type="r3d_18",
    ):
        super(DeerSingleStreamResnet, self).__init__()
        self.resnet, in_features = fine_tune_resnet(
            feature_extract, pretrained, resnet_type
        )
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc = nn.Linear(
            in_features=in_features, out_features=num_classes, bias=True
        )

    def forward(self, x, _):
        x = self.resnet(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class DeerSingleStreamMViT(nn.Module):
    def __init__(self, num_classes=2, feature_extract=False, pretrained=False):
        super(DeerSingleStreamMViT, self).__init__()
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
