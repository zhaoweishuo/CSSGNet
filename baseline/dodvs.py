import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.cv1 = ConvBNAct(channels, hidden, 1, 1)
        self.cv2 = ConvBNAct(hidden, channels, 3, 1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        hidden = out_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden * 2, 1, 1)
        self.blocks = nn.ModuleList(
            [Bottleneck(hidden) for _ in range(depth)]
        )
        self.cv2 = ConvBNAct(hidden * (2 + depth), out_channels, 1, 1)

    def forward(self, x):
        x1, x2 = self.cv1(x).chunk(2, dim=1)
        parts = [x1, x2]
        y = x2
        for block in self.blocks:
            y = block(y)
            parts.append(y)
        return self.cv2(torch.cat(parts, dim=1))


class SPPF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.cv1 = ConvBNAct(channels, hidden, 1, 1)
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.cv2 = ConvBNAct(hidden * 4, channels, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class OrientedDetectorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNAct(3, 32, 3, 2),
            ConvBNAct(32, 64, 3, 2),
            CSPBlock(64, 64, 1),
            ConvBNAct(64, 128, 3, 2),
            CSPBlock(128, 128, 2),
            ConvBNAct(128, 256, 3, 2),
            CSPBlock(256, 256, 2),
            ConvBNAct(256, 512, 3, 2),
            CSPBlock(512, 512, 1),
            SPPF(512)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.region_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 8)
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        r = self.region_head(f)
        box = torch.sigmoid(r[:, :4])
        quat = F.normalize(r[:, 4:8], p=2, dim=1, eps=1e-8)
        return f, torch.cat((box, quat), dim=1)


class DODVS(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = OrientedDetectorEncoder()
        self.pose_head = PoseHead(1040)

    def forward(self, x1, x2):
        f1, r1 = self.encoder(x1)
        f2, r2 = self.encoder(x2)
        x = torch.cat((f1, r1, f2, r2), dim=1)
        return self.pose_head(x)


def build_model(pretrained=False):
    return DODVS(pretrained=pretrained)
