# data/splits.py
from pathlib import Path
from typing import Literal, Tuple
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

    def get_query_gallery_from_df(
            self,
            df: pd.DataFrame,
            n_gallery: int = 1,
            seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        gallery_idx, query_idx = [], []
        for identity, group in df.groupby("identity"):
            n = min(n_gallery, len(group))
            sampled = group.sample(n=n, random_state=seed)
            gallery_idx.extend(sampled.index.tolist())
            query_idx.extend(group.drop(sampled.index).index.tolist())

        query_df = df.loc[query_idx].reset_index(drop=True)
        gallery_df = df.loc[gallery_idx].reset_index(drop=True)
        return query_df, gallery_df

    def get_val_test( #TIME SPLIT IS NOT OPEN SET, CODE NEEDS REVIEW FOR THIS CASE!
            self,
            split_type: SplitType,
            val_ratio: float = 0.5,
            seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_df = self.get_predefined(split_type, "test")
        unique_ids = test_df["identity"].drop_duplicates()
        val_ids = unique_ids.sample(frac=val_ratio, random_state=seed)

        val_df = test_df[test_df["identity"].isin(val_ids)].reset_index(drop=True)
        test_df = test_df[~test_df["identity"].isin(val_ids)].reset_index(drop=True)

        return val_df, test_df

    def get_train_train_calibrator( #TIME SPLIT IS NOT OPEN SET, CODE NEEDS REVIEW FOR THIS CASE!
            self,
            split_type: SplitType,
            train_ratio:float = 0.90,
            seed:int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_df = self.get_predefined(split_type, "train")
        unique_ids = test_df["identity"].drop_duplicates()
        val_ids = unique_ids.sample(frac=train_ratio, random_state=seed)

        train_df = test_df[test_df["identity"].isin(val_ids)].reset_index(drop=True)
        train_calibrator_df = test_df[~test_df["identity"].isin(val_ids)].reset_index(drop=True)

        return train_df, train_calibrator_df

