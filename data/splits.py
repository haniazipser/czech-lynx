# data/splits.py
from pathlib import Path
from typing import Literal
import pandas as pd
from wildlife_datasets import datasets, splits

SplitType = Literal["geo_aware", "time_open", "time_closed"]

SPLIT_COLUMN = {
    "geo_aware":  "split-geo_aware",
    "time_open":  "split-time_open",
    "time_closed": "split-time_closed",
}


class CzechLynxSplitter:
    def __init__(self, data_root: Path, csv_path: Path):
        df = pd.read_csv(csv_path).rename(columns={"unique_name": "identity"})
        self.dataset = datasets.CzechLynxv2(str(data_root), df)
        self.df = self.dataset.df

    def get_predefined(
        self,
        split_type: SplitType,
        role: Literal["train", "test"],
    ) -> pd.DataFrame:
        col = SPLIT_COLUMN[split_type]
        return self.df[self.df[col] == role].copy().reset_index(drop=True)

    def analyze(self, split_type: SplitType) -> None:
        train_df = self.get_predefined(split_type, "train")
        test_df  = self.get_predefined(split_type, "test")
        train_ids = set(train_df["identity"])
        test_ids  = set(test_df["identity"])
        print(f"\n=== {split_type} ===")
        print(f"  train: {len(train_df):>6} photos | {len(train_ids):>4} animals")
        print(f"  test:  {len(test_df):>6} photos | {len(test_ids):>4} animals")
        print(f"  overlap: {len(train_ids & test_ids)} | nly in test: {len(test_ids - train_ids)}")