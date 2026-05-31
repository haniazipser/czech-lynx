import argparse
import os

import torch
import wandb

import torch.nn as nn
from config.presets.baseline import get_config
from data.dataloader import CzechLynxDataModule
from data.transforms import get_transforms
from evaluation.retrieval import RetrievalEvaluator
from models.baseline import EfficientNetBaseline
from training.trainer import Trainer
from evaluation.xai.gradcam import run_gradcam_analysis
from evaluation.visualize import run_tsne_analysis, run_tsne_camera_vs_identity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None,
                        help="Existing wandb run ID — skips training, runs GradCAM only.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root path.")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Override CSV path.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint dir (e.g. Google Drive path).")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    if args.data_root:
        cfg.data_root = args.data_root
    if args.csv_path:
        cfg.csv_path = args.csv_path

    checkpoint_base = args.checkpoint_dir or "run"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dm = CzechLynxDataModule(cfg)
    model = EfficientNetBaseline(num_classes=dm.num_classes)
    run_id = args.run_id

    if run_id is None:
        print(f"Num classes (train): {dm.num_classes}")
        train_loader = dm.train_loader()
        query_loader = dm.query_loader()
        gallery_loader = dm.gallery_loader()

        run = wandb.init(
            entity="haniazipser2004-",
            project="czech-lynx",
            name=f"baseline-efficientnet-{cfg.split_type}",
            config=cfg.model_dump(),
        )
        run_id = run.id

        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_loader=train_loader,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            device=device,
            checkpoint_dir=f"{checkpoint_base}/{run_id}/checkpoints",
            criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
            evaluator=RetrievalEvaluator(device)
        )
        trainer.train(run)
        run.finish()

    print("\nRunning GradCAM...")
    checkpoint = f"{checkpoint_base}/{run_id}/checkpoints/best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    _, val_transform = get_transforms(cfg.experiment_type)

    output_dir = f"{checkpoint_base}/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    run_gradcam_analysis(
        model=model,
        dataset=dm.test_ds,
        device=device,
        val_transform=val_transform,
        save_path=f"{output_dir}/gradcam_{cfg.split_type}.png",
    )

    print("\nRunning t-SNE...")

    run_tsne_analysis(
        model=model,
        dataset=dm.test_ds,
        device=device,
        val_transform=val_transform,
        color_by=["identity", "trap_id"],
        save_path=f"{output_dir}/tsne.png"
    )

    run_tsne_camera_vs_identity(
        model=model,
        dataset=dm.test_ds,
        device=device,
        val_transform=val_transform,
        save_path=f"{output_dir}/tsne_camera_proof.png"
    )


if __name__ == "__main__":
    main()