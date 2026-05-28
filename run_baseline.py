from data.dataloader import CzechLynxDataModule
from config.presets.baseline import get_config

cfg = get_config()
dm = CzechLynxDataModule(cfg)
batch_imgs, batch_labels = next(iter(dm.train_loader()))
print(f"Batch shape: {batch_imgs.shape}")
print(f"Num classes: {dm.num_classes}")