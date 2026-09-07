from pathlib import Path
from thop import profile
import torch
from torch import nn
from torchvision import models


class Vs(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        weights_path: str = "./pretrained/vit_b_16-c867db91.pth",
    ):
        super().__init__()

        self.vit = models.vit_b_16()

        if pretrained:
            checkpoint_path = Path(weights_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Pretrained weights not found: {checkpoint_path}"
                )

            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.vit.load_state_dict(state_dict)


        self.out = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 6),
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:

        feature1 = self.vit(x1)
        feature2 = self.vit(x2)

        features = torch.cat((feature2, feature1), dim=1)
        pose = self.out(features)

        return pose


if __name__ == "__main__":
    net = Vs(pretrained=False).eval()

    input1 = torch.rand(1, 3, 224, 224)
    input2 = torch.rand(1, 3, 224, 224)

    actual_params = sum(p.numel() for p in net.parameters())

    with torch.no_grad():
        macs, thop_params = profile(
            net,
            inputs=(input1, input2),
            verbose=False,
        )

    print(f"Actual parameters: {actual_params:,}")
    print(f"Actual parameters: {actual_params / 1e6:.3f} M")
    print(f"THOP parameters: {thop_params / 1e6:.3f} M")
    print(f"THOP counted MACs: {macs / 1e9:.3f} G")