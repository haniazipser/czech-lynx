import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from models.base import BaseModel


class EfficientNetBaseline(BaseModel):

    def __init__(self, num_classes: int, freeze_backbone: bool = False):
        super().__init__()

        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        embedding_dim = backbone.classifier[1].in_features

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.embedding_dim = embedding_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(embedding_dim, num_classes),
        )

    @property
    def target_layer(self) -> nn.Module:
        return self.features[-1]

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.get_embeddings(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.get_embeddings(x), dim=1)