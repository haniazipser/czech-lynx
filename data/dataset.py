from pathlib import Path
from typing import Callable
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CzechLynxDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        transform: Callable | None = None,
    ):
        self.df = df
        self.data_root = data_root
        self.transform = transform

        unique = sorted(df["identity"].unique())
        self.label_map = {name: idx for idx, name in enumerate(unique)}
        self.num_classes = len(unique)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.data_root / row["path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_map[row["identity"]]
        return image, label

    def get_identity(self, idx: int) -> str:
        return self.df.iloc[idx]["identity"]