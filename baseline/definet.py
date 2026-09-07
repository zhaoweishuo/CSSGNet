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


class DEFINet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        vgg = models.vgg16(weights=weights)
        self.encoder = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose_head = PoseHead(512)

    def _encode(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        f1 = self._encode(x1)
        f2 = self._encode(x2)
        x = f2 - f1
        x = self.pool(x).flatten(1)
        return self.pose_head(x)


def build_model(pretrained=False):
    return DEFINet(pretrained=pretrained)
