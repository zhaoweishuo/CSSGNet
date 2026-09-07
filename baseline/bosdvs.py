import torch
import torch.nn as nn
from torchvision import models


class PoseHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 6)
        )

    def forward(self, x):
        return self.net(x)


class BroadLayer(nn.Module):
    def __init__(self, input_nodes=2000, feature_nodes=6, enhancement_nodes=8000):
        super().__init__()
        self.feature_maps = nn.ModuleList(
            [nn.Linear(input_nodes, feature_nodes) for _ in range(10)]
        )
        self.enhancement = nn.Linear(feature_nodes * 10, enhancement_nodes)

    def forward(self, x):
        features = torch.cat(
            [torch.sigmoid(layer(x)) for layer in self.feature_maps],
            dim=1
        )
        enhancement = torch.sigmoid(self.enhancement(features))
        return torch.cat((features, enhancement), dim=1)


class BOSDVS(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.encoder = models.mobilenet_v3_small(weights=weights)
        self.broad = BroadLayer(2000, 6, 8000)
        self.pose_head = PoseHead(8060)

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        x = torch.cat((f1, f2), dim=1)
        x = self.broad(x)
        return self.pose_head(x)


def build_model(pretrained=False):
    return BOSDVS(pretrained=pretrained)
