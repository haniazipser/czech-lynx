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

        train_df, train_calibrator_df = splitter.get_train_train_calibrator(cfg.split_type)
        val_df, test_df = splitter.get_val_test(cfg.split_type)

        # query/gallery for val set
        val_query_df, val_gallery_df = splitter.get_query_gallery_from_df(val_df)
        # query/gallery for test set
        test_query_df, test_gallery_df = splitter.get_query_gallery_from_df(test_df)

        #query/gallery for train calibrator set
        train_calibrator_query_df, train_calibrator_gallery_df = splitter.get_query_gallery_from_df(train_calibrator_df)

        train_transform, val_transform= get_transforms(cfg.experiment_type, cfg.image_size)

        self.train_ds = CzechLynxDataset(train_df, cfg.data_root, train_transform)
        self.train_calibrator_query_ds = CzechLynxDataset(train_calibrator_query_df, cfg.data_root, train_transform)
        self.train_calibrator_gallery_ds = CzechLynxDataset(train_calibrator_gallery_df, cfg.data_root, train_transform)
        self.val_ds = CzechLynxDataset(val_df, cfg.data_root, val_transform)
        self.test_ds  = CzechLynxDataset(test_df,  cfg.data_root, val_transform)
        self.val_query_ds = CzechLynxDataset(val_query_df, cfg.data_root, val_transform)
        self.val_gallery_ds = CzechLynxDataset(val_gallery_df, cfg.data_root, val_transform)
        self.test_query_ds = CzechLynxDataset(test_query_df, cfg.data_root, val_transform)
        self.test_gallery_ds = CzechLynxDataset(test_gallery_df, cfg.data_root, val_transform)

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

    def val_query_loader(self) -> DataLoader:
        return DataLoader(
            self.val_query_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_gallery_loader(self) -> DataLoader:
        return DataLoader(
            self.val_gallery_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def test_query_loader(self) -> DataLoader:
        return DataLoader(
            self.test_query_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def test_gallery_loader(self) -> DataLoader:
        return DataLoader(
            self.test_gallery_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )