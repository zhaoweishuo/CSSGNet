from torchvision import models
from torch import nn
import torch
from torchsummary import summary


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=3):
        super(CBAM, self).__init__()
        # 通道注意力机制
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel//reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction, out_features=in_channel, bias=False)
        )
        self.tanh = nn.Tanh()
        # 空间注意力机制
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)

    def forward(self, x):
        # 通道
        maxout = self.max_pool(x)
        maxout = self.mlp(maxout.view(maxout.size(0), -1))
        avgout = self.avg_pool(x)
        avgout = self.mlp(avgout.view(avgout.size(0), -1))
        channel_out = self.tanh(maxout+avgout)
        channel_out = channel_out.view(x.size(0),x.size(1),1,1)
        channel_out = channel_out*x

        # 空间
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        space_out = torch.cat((max_out, mean_out),dim=1)
        space_out = self.tanh(self.conv(space_out))
        space_out = x * space_out

        out = space_out*channel_out

        return out


class SeBlock(nn.Module):
    # 自定义的卷积块
    def __init__(self, in_channel, out_channel):
        super(SeBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.attention = CBAM(in_channel)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.attention(x)
        out = self.block1(out)
        out = out+x
        out = self.block2(out)
        return out


class Sal(nn.Module):

    def __init__(self, pretrained=True):
        super(Sal, self).__init__()
        resnet = models.resnet34()

        if pretrained:
            resnet.load_state_dict(torch.load("./pretrained/resnet34-b627a593.pth"))

        # 编码器
        self.resnet = resnet
        self.encoder0 = nn.Sequential(
            resnet.conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1,1,ceil_mode=False),
        )  # in 3*224*224 out 64*56*56
        self.encoder1 = resnet.layer1  # in 64*56*56 out 64*56*56
        self.encoder2 = resnet.layer2  # in 64*56*56 out 128*28*28
        self.encoder3 = resnet.layer3  # in 128*28*28 out 256*14*14
        self.encoder4 = resnet.layer4  # in 256*14*14 out 512*7*7

        # 减少通道一次
        self.mid = self.decoder4 = SeBlock(512, 512)  # 512*7*7

        # 解码器
        self.decoder4 = SeBlock(512, 256)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 256*14*14
        self.decoder3 = SeBlock(256, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 128*28*28
        self.decoder2 = SeBlock(128, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 64*56*56
        self.decoder1 = SeBlock(64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 64*112*112
        self.decoder0 = SeBlock(64, 32)
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 224*224
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        en0 = self.encoder0(x)
        en1 = self.encoder1(en0)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)

        mid = self.mid(en4)

        de4 = self.up4(self.decoder4(mid+en4))
        de3 = self.up3(self.decoder3(de4+en3))
        de2 = self.up2(self.decoder2(de3+en2))
        de1 = self.up1(self.decoder1(de2+en1))
        de0 = self.up0(self.decoder0(de1))
        out = self.out(de0)
        return out


if __name__ == '__main__':
    net = Sal(pretrained=False)
    # print(net)
    summary(net, input_size=(3, 224, 224), device="cpu")  # 不知道为什么，一维向量的输入需要多加上一个维度
