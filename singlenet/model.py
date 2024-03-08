import torch
from torch import nn


class SingleNet(nn.Module):
    def __init__(
            self,
            image_head: nn.Module,
            features_head: nn.Module,
            backbone: nn.Module,
            final_activation: nn.Module | None = None,
    ):
        super().__init__()

        self.image_head = image_head
        self.features_head = features_head
        self.backbone = backbone
        self.final_activation = final_activation

    def forward(self, x, features):
        x = self.image_head(x)
        features = self.features_head(features)

        x = self.backbone(torch.cat([x, features], dim=1))

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x