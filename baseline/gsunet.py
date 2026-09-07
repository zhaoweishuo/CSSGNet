import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, 1, 3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * 4, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * 4, channels, 1)

    def forward(self, x):
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        return x + y


class CHAG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.skip_path = ConvNeXtUnit(channels)
        self.gate_path = ConvNeXtUnit(channels)
        self.attention = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, skip, gate):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(
                gate,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        a = self.skip_path(skip) + self.gate_path(gate)
        return skip * self.attention(a)


class DecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )
        self.gate = CHAG(out_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1),
            LayerNorm2d(out_channels),
            nn.GELU(),
            ConvNeXtUnit(out_channels)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        skip = self.gate(skip, x)
        return self.fuse(torch.cat((x, skip), dim=1))


class GSFeatureEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        f = backbone.features
        self.stem = f[0]
        self.stage1 = f[1]
        self.down1 = f[2]
        self.stage2 = f[3]
        self.down2 = f[4]
        self.stage3 = f[5]
        self.down3 = f[6]
        self.stage4 = f[7]
        self.dec3 = DecoderStage(768, 384)
        self.dec2 = DecoderStage(384, 192)
        self.dec1 = DecoderStage(192, 96)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        s1 = self.stage1(x)
        x = self.down1(s1)
        s2 = self.stage2(x)
        x = self.down2(s2)
        s3 = self.stage3(x)
        x = self.down3(s3)
        s4 = self.stage4(x)
        x = self.dec3(s4, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.pool(x).flatten(1)


class GSUNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = GSFeatureEncoder(pretrained=pretrained)
        self.pose_head = PoseHead(192)

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        return self.pose_head(torch.cat((f1, f2), dim=1))


def build_model(pretrained=False):
    return GSUNet(pretrained=pretrained)
