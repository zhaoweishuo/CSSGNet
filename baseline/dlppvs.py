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


class DLPPVS(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = models.ResNet152_Weights.DEFAULT if pretrained else None
        backbone = models.resnet152(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.pose_head = PoseHead(4096)

    def _encode(self, x):
        x = self.encoder(x)
        return torch.flatten(x, 1)

    def forward(self, x1, x2):
        f1 = self._encode(x1)
        f2 = self._encode(x2)
        return self.pose_head(torch.cat((f1, f2), dim=1))


def build_model(pretrained=False):
    return DLPPVS(pretrained=pretrained)
