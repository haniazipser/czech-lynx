from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from data.dataset import CzechLynxDataset
from data.splits import CzechLynxSplitter
from data.transforms import get_transforms
from config.config import DataConfig




class CzechLynxDataModule:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

        splitter = CzechLynxSplitter(cfg.data_root, cfg.csv_path)

        train_df = splitter.get_predefined(cfg.split_type, "train")
        test_df = splitter.get_predefined(cfg.split_type, "test")
        query_df, gallery_df = splitter.get_query_gallery(cfg.split_type)

        train_transform, val_transform= get_transforms(cfg.experiment_type)

        self.train_ds = CzechLynxDataset(train_df, cfg.data_root, train_transform)
        self.test_ds  = CzechLynxDataset(test_df,  cfg.data_root, val_transform)
        self.query_ds = CzechLynxDataset(query_df, cfg.data_root, val_transform)
        self.gallery_ds = CzechLynxDataset(gallery_df, cfg.data_root, val_transform)

    @property
    def num_classes(self) -> int:
        return self.train_ds.num_classes

    def train_loader(self) -> DataLoader:
        labels = list(self.train_ds.df["identity"].map(self.train_ds.label_map))
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def query_loader(self) -> DataLoader:
        return DataLoader(
            self.query_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def gallery_loader(self) -> DataLoader:
        return DataLoader(
            self.gallery_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )