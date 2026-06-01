from datetime import datetime
import os
import torch
import torch.nn as nn
from timm.optim import optimizer_kwargs
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

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
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        checkpoint_dir: str = "checkpoints",
        track_accuracy: bool = False,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.evaluator = evaluator
        self.track_accuracy = track_accuracy
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _run_epoch(self) -> tuple[float, float | None]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            output = self.model(imgs)
            loss = self.criterion(output, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            if self.track_accuracy:
                correct += (output.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        acc = correct / total if self.track_accuracy else None
        return total_loss / total, acc

    def train(self, run: wandb.sdk.wandb_run.Run, checkpoint_every: int = 5):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_rank1 = 0.0

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_acc = self._run_epoch()
            self.scheduler.step()
            metrics = self.evaluator.evaluate(
                self.model, self.query_loader, self.gallery_loader
            )

            log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "eval/rank1": metrics["rank1"],
                "eval/rank5": metrics["rank5"],
                "eval/map": metrics["map"],
                "lr": self.scheduler.get_last_lr()[0],
            }
            if train_acc is not None:
                log["train/acc"] = train_acc

            run.log(log)

            acc_str = f" acc {train_acc:.3f} |" if train_acc is not None else ""
            print(
                f"{datetime.now().strftime('%H:%M:%S')} | "
                f"Epoch {epoch:>3}/{self.cfg.epochs} | "
                f"train loss {train_loss:.4f} |{acc_str} "
                f"Rank-1 {metrics['rank1']:.4f} mAP {metrics['map']:.4f}"
            )

            if metrics["rank1"] > best_rank1:
                best_rank1 = metrics["rank1"]
                torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/best.pt")
                print(f"  -> new best Rank-1 ({best_rank1:.4f})")

            if epoch % checkpoint_every == 0:
                torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/epoch_{epoch:03d}.pt")

        torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/final.pt")
        print(f"\nBest Rank-1: {best_rank1:.4f}")

        cmc_data = [[k + 1, v] for k, v in enumerate(metrics["cmc"])]
        run.log({
            "eval/cmc_curve": wandb.plot.line(
                wandb.Table(data=cmc_data, columns=["rank", "accuracy"]),
                x="rank",
                y="accuracy",
                title="CMC Curve",
            )
        })

        return best_rank1