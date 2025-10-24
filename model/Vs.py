from torchvision import models
from torch import nn
import torch


class Vs(nn.Module):

    def __init__(self, pretrained=True):
        super(Vs, self).__init__()
        vit1 = models.vit_b_16()
        vit2 = models.vit_b_16()
        if pretrained:
            vit1.load_state_dict(torch.load("./pretrained/vit_b_16-c867db91.pth"))
            vit2.load_state_dict(torch.load("./pretrained/vit_b_16-c867db91.pth"))

        # 当前图像流
        self.vit1 = vit1
        self.vit2 = vit2

        self.out = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 6),
        )

    def forward(self, x1, x2):
        x1 = self.vit1(x1)
        x2 = self.vit2(x2)
        y = torch.concat((x2, x1), dim=1)
        y = self.out(y)
        return y


if __name__ == '__main__':
    net = Vs(pretrained=False)
    print(net)
    in1 = torch.rand(2, 3, 224, 224)
    in2 = torch.rand(2, 3, 224, 224)
    out = net(in1, in2)
    print(out.shape)


