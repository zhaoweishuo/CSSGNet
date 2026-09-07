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


class VGGEmbedding(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        vgg = models.vgg16(weights=weights)
        self.features = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class SVIVS(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.observation_encoder = VGGEmbedding(pretrained=pretrained)
        self.query_encoder = VGGEmbedding(pretrained=pretrained)
        self.action_embedding = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True)
        )
        self.recurrent = nn.LSTM(
            input_size=512 + 512 + 64,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        self.pose_head = PoseHead(512)

    def forward(self, x1, x2):
        observation = self.observation_encoder(x1)
        query = self.query_encoder(x2)
        previous_action = torch.zeros(
            x1.shape[0],
            3,
            dtype=x1.dtype,
            device=x1.device
        )
        action = self.action_embedding(previous_action)
        x = torch.cat((observation, query, action), dim=1).unsqueeze(1)
        x, _ = self.recurrent(x)
        return self.pose_head(x[:, -1])


def build_model(pretrained=False):
    return SVIVS(pretrained=pretrained)
