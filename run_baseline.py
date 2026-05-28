from data.dataloader import CzechLynxDataModule
from config.presets.baseline import get_config
import pandas as pd

cfg = get_config()
dm = CzechLynxDataModule(cfg)
batch_imgs, batch_labels = next(iter(dm.train_loader()))
print(f"Batch shape: {batch_imgs.shape}")
print(f"Num classes: {dm.num_classes}")

# from wildlife_datasets import datasets
# from pathlib import Path
#
# df = pd.read_csv("data/kaggle-data/CzechLynxDataset-Metadata-Real.csv")
# df = df.rename(columns={"unique_name": "identity"})
#
# d = datasets.CzechLynxv2("data/kaggle-data", df)
# print(d.df.columns.tolist())
# print(d.df.head(2))