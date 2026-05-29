import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from config.config import DataConfig
from evaluation.retrieval import RetrievalEvaluator
from models.base import BaseModel


class Trainer:

    def __init__(
        self,
        model: BaseModel,
        cfg: DataConfig,
        train_loader: DataLoader,
        query_loader: DataLoader,
        gallery_loader: DataLoader,
        device: torch.device,
        criterion: nn.Module,
        evaluator: RetrievalEvaluator,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.criterion = criterion
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        self.evaluator = evaluator

    def _run_epoch(self, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for imgs, labels in self.train_loader:
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
        best_rank1 = 0.0

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_acc = self._run_epoch(train=True)
            self.scheduler.step()

            metrics = self.evaluator.evaluate(
                self.model, self.query_loader, self.gallery_loader
            )

            run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "eval/rank1": metrics["rank1"],
                "eval/rank5": metrics["rank5"],
                "eval/map": metrics["map"],
                "lr": self.scheduler.get_last_lr()[0],
            })

            print(
                f"Epoch {epoch:>3}/{self.cfg.epochs} | "
                f"loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"Rank-1 {metrics['rank1']:.4f} mAP {metrics['map']:.4f}"
            )

            if metrics["rank1"] > best_rank1:
                best_rank1 = metrics["rank1"]
                torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/best.pt")
                print(f"  -> new best Rank-1 ({best_rank1:.4f})")

            if epoch % checkpoint_every == 0:
                torch.save(self.model.state_dict(), f"{checkpoint_dir}/epoch_{epoch:03d}.pt")

        torch.save(self.model.state_dict(), f"{checkpoint_dir}/final.pt")
        print(f"\nBest Rank-1: {best_rank1:.4f}")

        # Final eval with CMC
        cmc_data = [[k + 1, v] for k, v in enumerate(metrics["cmc"])]
        run.log({
            "eval/cmc_curve": wandb.plot.line(
                wandb.Table(data=cmc_data, columns=["rank", "accuracy"]),
                x="rank",
                y="accuracy",
                title="CMC Curve — Baseline",
            )
        })

        return best_rank1