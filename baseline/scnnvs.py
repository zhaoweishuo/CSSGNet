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


class SCNNVS(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        alexnet = models.alexnet(weights=weights)
        self.features = alexnet.features
        self.channel_reduce = nn.Conv2d(256, 96, 1)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.match = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(96 * 6 * 6 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024)
        )
        self.pose_head = PoseHead(1024)

    def _encode(self, x):
        x = self.features(x)
        x = self.channel_reduce(x)
        x = self.pool(x)
        return x.flatten(1)

    def forward(self, x1, x2):
        f1 = self._encode(x1)
        f2 = self._encode(x2)
        x = torch.cat((f1, f2), dim=1)
        x = self.match(x)
        return self.pose_head(x)


def build_model(pretrained=False):
    return SCNNVS(pretrained=pretrained)
