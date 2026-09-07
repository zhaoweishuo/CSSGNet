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


class ECNVS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, 7, 2, 3)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 2)
        self.conv3 = nn.Conv2d(128, 256, 5, 2, 2)
        self.conv3_1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv4_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5 = nn.Conv2d(512, 512, 3, 2, 1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 1024, 3, 2, 1)
        self.conv6_1 = nn.Conv2d(1024, 1024, 3, 1, 1)

        self.predict_flow6 = nn.Conv2d(1024, 2, 3, 1, 1)
        self.deconv5 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.upflow6 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        self.predict_flow5 = nn.Conv2d(1026, 2, 3, 1, 1)
        self.deconv4 = nn.ConvTranspose2d(1026, 256, 4, 2, 1)
        self.upflow5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        self.predict_flow4 = nn.Conv2d(770, 2, 3, 1, 1)
        self.deconv3 = nn.ConvTranspose2d(770, 128, 4, 2, 1)
        self.upflow4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        self.predict_flow3 = nn.Conv2d(386, 2, 3, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(386, 64, 4, 2, 1)
        self.upflow3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)

        self.predict_flow2 = nn.Conv2d(194, 2, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((56, 56))
        self.fc7 = nn.Sequential(
            nn.Linear(2 * 56 * 56, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.pose_head = PoseHead(4096)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def _resize(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)

        c1 = self.act(self.conv1(x))
        c2 = self.act(self.conv2(c1))
        c3 = self.act(self.conv3(c2))
        c3_1 = self.act(self.conv3_1(c3))
        c4 = self.act(self.conv4(c3_1))
        c4_1 = self.act(self.conv4_1(c4))
        c5 = self.act(self.conv5(c4_1))
        c5_1 = self.act(self.conv5_1(c5))
        c6 = self.act(self.conv6(c5_1))
        c6_1 = self.act(self.conv6_1(c6))

        f6 = self.predict_flow6(c6_1)
        d5 = self._resize(self.act(self.deconv5(c6_1)), c5_1)
        u6 = self._resize(self.upflow6(f6), c5_1)
        cat5 = torch.cat((c5_1, d5, u6), dim=1)

        f5 = self.predict_flow5(cat5)
        d4 = self._resize(self.act(self.deconv4(cat5)), c4_1)
        u5 = self._resize(self.upflow5(f5), c4_1)
        cat4 = torch.cat((c4_1, d4, u5), dim=1)

        f4 = self.predict_flow4(cat4)
        d3 = self._resize(self.act(self.deconv3(cat4)), c3_1)
        u4 = self._resize(self.upflow4(f4), c3_1)
        cat3 = torch.cat((c3_1, d3, u4), dim=1)

        f3 = self.predict_flow3(cat3)
        d2 = self._resize(self.act(self.deconv2(cat3)), c2)
        u3 = self._resize(self.upflow3(f3), c2)
        cat2 = torch.cat((c2, d2, u3), dim=1)

        f2 = F.relu(self.predict_flow2(cat2), inplace=True)
        f2 = self.pool(f2).flatten(1)
        f2 = self.fc7(f2)
        return self.pose_head(f2)


def build_model(pretrained=False):
    return ECNVS()
