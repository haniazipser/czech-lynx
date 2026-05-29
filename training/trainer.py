import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from config.config import DataConfig
from models.base import BaseModel


class Trainer:

    def __init__(
        self,
        model: BaseModel,
        cfg: DataConfig,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += imgs.size(0)

        return total_loss / total, correct / total

    def train(
        self,
        run: wandb.sdk.wandb_run.Run,
        checkpoint_every: int = 5,
    ):
        checkpoint_dir = self.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_acc = 0.0

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.test_loader, train=False)
            self.scheduler.step()

            run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": self.scheduler.get_last_lr()[0],
            })

            print(
                f"Epoch {epoch:>3}/{self.cfg.epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f"{checkpoint_dir}/best.pt")
                print(f"  -> new best ({best_acc:.4f})")

            if epoch % checkpoint_every == 0:
                torch.save(self.model.state_dict(), f"{checkpoint_dir}/epoch_{epoch:03d}.pt")

        torch.save(self.model.state_dict(), f"{checkpoint_dir}/final.pt")
        print(f"\nBest val acc: {best_acc:.4f}")
        return best_acc