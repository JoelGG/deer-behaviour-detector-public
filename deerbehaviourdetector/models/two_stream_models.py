import torch
import torch.nn as nn

from deerbehaviourdetector.models.backbones import fine_tune_resnet


class TwoStreamModelStackFuse(nn.Module):
    def __init__(
        self,
        num_classes=8,
        feature_extract=False,
        pretrained=True,
        resnet_type_rgb="r3d_18",
        resnet_type_flow="r3d_18",
    ):
        super(TwoStreamModelStackFuse, self).__init__()
        self.resnet_spacial, spacial_in_features = fine_tune_resnet(
            feature_extract, pretrained, resnet_type_rgb
        )
        self.resnet_temporal, temporal_in_features = fine_tune_resnet(
            feature_extract, pretrained, resnet_type_flow
        )

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self.fc = nn.Linear(
            in_features=spacial_in_features + temporal_in_features,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, spacial_input, temporal_input):
        x_spacial = self.resnet_spacial(spacial_input)
        x_temporal = self.resnet_temporal(temporal_input)

        x = torch.cat((x_spacial, x_temporal), dim=1)
        x = self.avgpool(x_spacial)
        x = self.fc(x)
        return x


class TwoStreamModelAdaptiveFuse(nn.Module):
    def __init__(
        self,
        num_classes=8,
        feature_extract=False,
        pretrained=True,
        resnet_type_rgb="r3d_18",
        resnet_type_flow="r3d_18",
    ):
        super(TwoStreamModelAdaptiveFuse, self).__init__()
        self.resnet_spacial, spacial_in_features = fine_tune_resnet(
            feature_extract, pretrained, resnet_type_rgb
        )
        self.resnet_temporal, temporal_in_features = fine_tune_resnet(
            feature_extract, pretrained, resnet_type_flow
        )

        self.avgpool_spacial = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.avgpool_temporal = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self.fc = nn.Linear(
            in_features=spacial_in_features + temporal_in_features,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, spacial_input, temporal_input):
        x_spacial = self.resnet_spacial(spacial_input)
        x_temporal = self.resnet_temporal(temporal_input)

        x_spacial = self.avgpool_spacial(x_spacial)
        x_temporal = self.avgpool_temporal(x_temporal)

        x = torch.cat((x_spacial, x_temporal), dim=1)
        x = self.fc(x)

        return x


class TwoStreamModelAvgFuse(nn.Module):
    def __init__(
        self,
        num_classes=8,
        feature_extract=False,
        pretrained=True,
        resnet_type_rgb="r3d_18",
        resnet_type_flow="r3d_18",
    ):
        super(TwoStreamModelAvgFuse, self).__init__()
        self.resnet_spacial, spacial_in_features = fine_tune_resnet(
            feature_extract, pretrained, resnet_type_rgb
        )
        self.resnet_temporal, temporal_in_features = fine_tune_resnet(
            feature_extract, pretrained, resnet_type_flow
        )

        self.avgpool_spacial = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.avgpool_temporal = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self.fc_spacial = nn.Linear(
            in_features=spacial_in_features,
            out_features=num_classes,
            bias=True,
        )
        self.fc_temporal = nn.Linear(
            in_features=temporal_in_features,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, spacial_input, temporal_input):
        x_spacial = self.resnet_spacial(spacial_input)
        x_temporal = self.resnet_temporal(temporal_input)

        x_spacial = self.avgpool_spacial(x_spacial)
        x_temporal = self.avgpool_temporal(x_temporal)

        x_spacial = self.fc_spacial(x_spacial)
        x_temporal = self.fc_temporal(x_temporal)

        output = []
        for i in range(0, len(x_spacial)):
            output.append(torch.stack([x_spacial[i], x_temporal[i]]))

        x = torch.stack(output)
        x = torch.mean(output, dim=1)
        return x
