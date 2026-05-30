import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from models.base import BaseModel


class MegaDescriptorModel(BaseModel):

    def __init__(self, embedding_dim: int = 512, freeze_backbone: bool = False):
        super().__init__()

        self.backbone = create_model(
            "hf-hub:BVRA/MegaDescriptor-L-384",
            pretrained=True,
            num_classes=0,
        )
        backbone_dim = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim

    @property
    def target_layer(self) -> nn.Module:
        return self.backbone.blocks[-1]

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_embeddings(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.get_embeddings(x), dim=1)