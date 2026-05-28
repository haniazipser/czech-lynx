import torch
import wandb

from config.presets.baseline import get_config
from data.dataloader import CzechLynxDataModule
from data.transforms import get_transforms
from models.baseline import EfficientNetBaseline
from training.trainer import Trainer
from xai.gradcam import run_gradcam_analysis


def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    dm = CzechLynxDataModule(cfg)
    print(f"Num classes (train): {dm.num_classes}")

    train_loader = dm.train_loader()
    test_loader = dm.test_loader()


    model = EfficientNetBaseline(num_classes=dm.num_classes)


    run = wandb.init(
        entity="haniazipser2004-",
        project="czech-lynx",
        name=f"baseline-efficientnet-{cfg.split_type}",
        config=cfg.model_dump(),
    )

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
    )
    trainer.train(run)
    run.finish()

    # --- GradCAM ---
    print("\nGenerowanie GradCAM...")
    model.load_state_dict(torch.load("best_baseline.pt", map_location=device))
    _, val_transform = get_transforms(cfg.experiment_type)

    run_gradcam_analysis(
        model=model,
        dataset=dm.test_ds,
        device=device,
        val_transform=val_transform,
        num_samples=12,
        save_path=f"gradcam_{cfg.split_type}.png",
    )


if __name__ == "__main__":
    main()