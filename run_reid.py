import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import wandb

from config.presets.megadesc import get_config
from data.dataloader import CzechLynxDataModule
from data.transforms import get_transforms
from evaluation.retrieval import RetrievalEvaluator
from models.reid import MegaDescriptorModel
from models.losses import ArcFaceLoss
from training.trainer import Trainer
from evaluation.xai.gradcam import run_gradcam_analysis
from evaluation.visualize import run_tsne_analysis, run_tsne_camera_vs_identity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None,
                        help="Existing wandb run ID — skips training, runs analysis only.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dm = CzechLynxDataModule(cfg)
    model = MegaDescriptorModel()
    run_id = args.run_id

    if run_id is None:
        print(f"Num classes (train): {dm.num_classes}")
        train_loader = dm.train_loader()
        query_loader = dm.query_loader()
        gallery_loader = dm.gallery_loader()

        run = wandb.init(
            entity="haniazipser2004-",
            project="czech-lynx",
            name=f"reid-megadescriptor-{cfg.split_type}",
            config=cfg.model_dump(),
        )
        run_id = run.id

        criterion = ArcFaceLoss(
            embedding_dim=model.embedding_dim,
            num_classes=dm.num_classes,
        ).to(device)

        optimizer = AdamW([
            # Backbone
            {
                "params": filter(lambda p: p.requires_grad, model.backbone.parameters()),
                "lr": 3e-5
            },
            # ArcFace
            {
                "params": list(filter(lambda p: p.requires_grad, model.head.parameters())) +
                          list(filter(lambda p: p.requires_grad, criterion.parameters())),
                "lr": 3e-4
            }
        ], weight_decay=cfg.weight_decay)

        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=cfg.warmup_start,
            end_factor=cfg.warmup_end,
            total_iters=cfg.warmup_epochs
        )
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=(cfg.epochs - cfg.warmup_epochs)
        )

        mega_desc_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[cfg.warmup_epochs]
        )

        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_loader=train_loader,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            device=device,
            checkpoint_dir=f"run/{run_id}/checkpoints",
            criterion=criterion,
            evaluator=RetrievalEvaluator(device),
            optimizer=optimizer,
            scheduler=mega_desc_scheduler
        )
        trainer.train(run)
        run.finish()

    checkpoint = f"run/{run_id}/checkpoints/best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    _, val_transform = get_transforms(cfg.experiment_type)

    print("\nRunning t-SNE...")
    run_tsne_analysis(
        model=model,
        dataset=dm.test_ds,
        device=device,
        val_transform=val_transform,
        color_by=["identity", "trap_id"],
        save_path=f"run/{run_id}/tsne.png",
    )

    run_tsne_camera_vs_identity(
        model=model,
        dataset=dm.test_ds,
        device=device,
        val_transform=val_transform,
        save_path=f"run/{run_id}/tsne_camera_proof.png",
    )


if __name__ == "__main__":
    main()