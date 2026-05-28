from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def target_layer(self) -> nn.Module:
        """Layer to hook for GradCAM — each subclass defines its own."""
        ...