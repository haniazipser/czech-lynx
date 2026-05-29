import argparse
import torch
import wandb

import torch.nn as nn
from config.presets.baseline import get_config
from data.dataloader import CzechLynxDataModule
from data.transforms import get_transforms
from models.baseline import EfficientNetBaseline
from training.trainer import Trainer
from xai.gradcam import run_gradcam_analysis


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None,
                        help="Existing wandb run ID — skips training, runs GradCAM only.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dm = CzechLynxDataModule(cfg)
    model = EfficientNetBaseline(num_classes=dm.num_classes)

    run_id = args.run_id

    if run_id is None:
        print(f"Num classes (train): {dm.num_classes}")
        train_loader = dm.train_loader()
        test_loader = dm.test_loader()

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
            test_loader=test_loader,
            device=device,
            checkpoint_dir=f"run/{run_id}/checkpoints",
            criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
        )
        trainer.train(run)
        run.finish()

    print("\nRunning GradCAM...")
    checkpoint = f"run/{run_id}/checkpoints/best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, val_transform = get_transforms(cfg.experiment_type)

    run_gradcam_analysis(
        model=model,
        dataset=dm.test_ds,
        device=device,
        val_transform=val_transform,
        save_path=f"run/{run_id}/gradcam_{cfg.split_type}.png",
    )


if __name__ == "__main__":
    main()