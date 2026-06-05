import argparse
import os

import numpy as np
import torch
from pytorch_metric_learning import losses
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

import wandb

from config.presets.megadesc import get_config
from data.dataset import CzechLynxDataset
from data.splits import CzechLynxSplitter
from data.transforms import get_transforms
from evaluation.retrieval import RetrievalEvaluator
from models.reid import MegaDescriptorModel
from training.trainer import Trainer


def make_train_loader(dataset, batch_size, num_workers):
    labels = list(dataset.df["identity"].map(dataset.label_map))
    class_counts = torch.tensor(np.bincount(labels), dtype=torch.float32)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_eval_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_stage_data(cfg, train_root, train_csv, val_root, val_csv):
    synthetic_splitter = CzechLynxSplitter(train_root, train_csv)
    real_splitter = CzechLynxSplitter(val_root, val_csv)

    train_df, _ = synthetic_splitter.get_train_train_calibrator(cfg.split_type)
    val_df, _ = real_splitter.get_val_test(cfg.split_type)

    val_query_df, val_gallery_df = real_splitter.get_query_gallery_from_df(val_df)

    train_transform, val_transform = get_transforms(cfg.experiment_type, cfg.image_size)

    train_ds = CzechLynxDataset(train_df, train_root, train_transform)
    val_query_ds = CzechLynxDataset(val_query_df, val_root, val_transform)
    val_gallery_ds = CzechLynxDataset(val_gallery_df, val_root, val_transform)

    return train_ds, val_query_ds, val_gallery_ds


def train_stage(cfg, device, train_root, train_csv, val_root, val_csv, run_name, resume_from=None):
    train_ds, val_query_ds, val_gallery_ds = build_stage_data(cfg, train_root, train_csv, val_root, val_csv)
    train_loader = make_train_loader(train_ds, cfg.batch_size, cfg.num_workers)
    val_query_loader = make_eval_loader(val_query_ds, cfg.batch_size, cfg.num_workers)
    val_gallery_loader = make_eval_loader(val_gallery_ds, cfg.batch_size, cfg.num_workers)

    model = MegaDescriptorModel()
    if resume_from is not None:
        model.load_state_dict(torch.load(resume_from, map_location=device))
    model = model.to(device)

    criterion = losses.ArcFaceLoss(
        num_classes=len(train_ds.label_map),
        embedding_size=model.embedding_dim,
        margin=cfg.arcface_m,
        scale=cfg.arcface_s,
    ).to(device)

    optimizer = AdamW([
        {
            "params": filter(lambda p: p.requires_grad, model.backbone.parameters()),
            "lr": cfg.lr,
        },
        {
            "params": list(filter(lambda p: p.requires_grad, model.head.parameters())) + list(filter(lambda p: p.requires_grad, criterion.parameters())),
            "lr": cfg.lr,
        },
    ], weight_decay=cfg.weight_decay)

    scheduler_warmup = LinearLR(optimizer, start_factor=cfg.warmup_start, end_factor=cfg.warmup_end, total_iters=cfg.warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(cfg.epochs - cfg.warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[cfg.warmup_epochs])

    run = wandb.init(entity="haniazipser2004-", project="czech-lynx", name=run_name, config=cfg.model_dump())
    checkpoint_dir = f"run/{run.id}/checkpoints"
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        query_loader=val_query_loader,
        gallery_loader=val_gallery_loader,
        device=device,
        criterion=criterion,
        evaluator=RetrievalEvaluator(device),
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.train(run)
    run.finish()

    return os.path.join(checkpoint_dir, "best.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential synthetic pretraining and real fine-tuning for MegaDescriptor.")
    parser.add_argument("--pretrain-name", type=str, default="reid-megadescriptor-synthetic-pretrain")
    parser.add_argument("--finetune-name", type=str, default="reid-megadescriptor-real-finetune")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip synthetic pretraining and only run real fine-tuning.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    cfg.split_type = "geo_aware"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    synthetic_root = cfg.synthetic_data_root
    synthetic_csv = cfg.synthetic_csv_path
    real_root = cfg.data_root
    real_csv = cfg.csv_path

    checkpoint = None
    if not args.skip_pretrain:
        print("Stage 1: Pretraining on synthetic train and real validation...")
        checkpoint = train_stage(
            cfg,
            device,
            train_root=synthetic_root,
            train_csv=synthetic_csv,
            val_root=real_root,
            val_csv=real_csv,
            run_name=args.pretrain_name,
        )

    print("Stage 2: Fine-tuning on real train and real validation...")
    train_stage(
        cfg,
        device,
        train_root=real_root,
        train_csv=real_csv,
        val_root=real_root,
        val_csv=real_csv,
        run_name=args.finetune_name,
        resume_from=checkpoint,
    )


if __name__ == "__main__":
    main()
