import argparse
from pathlib import Path

import pandas as pd

from config.presets.megadesc import get_config
from data.splits import CzechLynxSplitter


def describe_identity_counts(name: str, df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("identity").size().sort_values(ascending=False)
    print(f"\n{name} set")
    print("--------------------------")
    print(f" rows       : {len(df)}")
    print(f" ids        : {counts.size}")
    print(f" min images : {counts.min():.0f}")
    print(f" max images : {counts.max():.0f}")
    print(f" mean images: {counts.mean():.2f}")
    print(f" median     : {counts.median():.2f}")
    print(f" std dev    : {counts.std():.2f}")
    print(f" <= 10 imgs : {(counts <= 10).sum()} / {counts.size}")
    print(f" <= 20 imgs : {(counts <= 20).sum()} / {counts.size}")
    print(f" > 200 imgs : {(counts > 200).sum()} / {counts.size}")
    print(f" > 500 imgs : {(counts > 500).sum()} / {counts.size}")
    print(f" >1000 imgs : {(counts > 1000).sum()} / {counts.size}")
    print(" top 10 identities:")
    for identity, count in counts.head(10).items():
        print(f"  {identity}: {count}")
    return counts


def describe_query_gallery(name: str, df: pd.DataFrame, splitter: CzechLynxSplitter):
    query_df, gallery_df = splitter.get_query_gallery_from_df(df)
    print(f"\n{name} query/gallery")
    print("--------------------------")
    print(f" query rows  : {len(query_df)}")
    print(f" gallery rows: {len(gallery_df)}")
    print(f" gallery ids : {gallery_df['identity'].nunique()}")
    print(f" query ids   : {query_df['identity'].nunique()}")
    if len(query_df) > 0:
        print(f" mean query images per identity: {query_df.groupby('identity').size().mean():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore validation/test split statistics for Czech Lynx retrieval."
    )
    parser.add_argument("--split-type", type=str, default=None,
                        help="Split type to analyze (time_open, time_closed, geo_aware).")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Dataset root folder. Defaults to config value.")
    parser.add_argument("--csv-path", type=Path, default=None,
                        help="Metadata CSV path. Defaults to config value.")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save per-identity count tables for val/test to CSV files.")
    args = parser.parse_args()

    cfg = get_config()
    split_type = args.split_type or cfg.split_type
    data_root = args.data_root or cfg.data_root
    csv_path = args.csv_path or cfg.csv_path

    print(f"Using split type: {split_type}")
    print(f"Data root: {data_root}")
    print(f"CSV path: {csv_path}")

    splitter = CzechLynxSplitter(data_root, csv_path)
    val_df, test_df = splitter.get_val_test(split_type)

    val_counts = describe_identity_counts("Validation", val_df)
    test_counts = describe_identity_counts("Test", test_df)
    describe_query_gallery("Validation", val_df, splitter)
    describe_query_gallery("Test", test_df, splitter)

    if args.save_csv:
        val_counts.to_frame(name="count").to_csv("explore_test_val_identity_counts.csv")
        test_counts.to_frame(name="count").to_csv("explore_test_test_identity_counts.csv")
        print("\nSaved per-identity count CSVs:")
        print(" - explore_test_val_identity_counts.csv")
        print(" - explore_test_test_identity_counts.csv")


if __name__ == "__main__":
    main()
