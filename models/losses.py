import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):

    def __init__(self, embedding_dim: int, num_classes: int, s: float, m: float):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.threshold = torch.cos(torch.tensor(torch.pi - m))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)

        cos_theta = torch.mm(embeddings, weight.t()).clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(1e-7, 1.0))

        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        cos_theta_m = torch.where(
            cos_theta > self.threshold.to(embeddings.device),
            cos_theta_m,
            cos_theta - self.sin_m * self.m,
        )

        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.unsqueeze(1), 1.0)
        logits = self.s * (one_hot * cos_theta_m + (1 - one_hot) * cos_theta)

        return F.cross_entropy(logits, labels)