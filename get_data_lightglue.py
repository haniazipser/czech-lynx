import json
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from datasets import tqdm
from lightglue.utils import load_image

from config.presets.megadesc import get_config
from data.dataloader import CzechLynxDataModule
from wildfusion.wildfusion import LocalMatcher
import numpy as np

import pycocotools.mask as mask_utils

def get_calibration_subset(df, n_per_identity=3, max_total=200):
    frames = []
    for identity, group in df.groupby('identity'):
        frames.append(group.sample(n=min(len(group), n_per_identity), random_state=42))
    subset = pd.concat(frames).reset_index(drop=True)
    if len(subset) > max_total:
        subset = subset.sample(n=max_total, random_state=42).reset_index(drop=True)
    return subset

def load_and_crop_via_rle(row, base_path, device, target_size=512):
    try:
        img_path = base_path / row['path']
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size


        mask_data = json.loads(row['mask'])
        mask = mask_utils.decode(mask_data).astype(np.uint8)

        pos = np.where(mask > 0)
        if len(pos[0]) == 0 or len(pos[1]) == 0:
            print(f"[FALLBACK] Mask is empty. Loading full image")
            tmp_path = Path("tmp_crop.jpg")
            img.save(tmp_path)
            return load_image(tmp_path, resize=target_size).to(device)

        ymin, ymax = np.min(pos[0]), np.max(pos[0])
        xmin, xmax = np.min(pos[1]), np.max(pos[1])

        xmin = max(0, int(xmin - 5))
        ymin = max(0, int(ymin - 5))
        xmax = min(img_w, int(xmax + 5))
        ymax = min(img_h, int(ymax + 5))

        if (xmax - xmin) <= 5 or (ymax - ymin) <= 5:
            print(f"[FALLBACK] Box too small. Loading full image")
            tmp_path = Path("tmp_crop.jpg")
            img.save(tmp_path)
            return load_image(tmp_path, resize=target_size).to(device)

        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        tmp_path = Path("tmp_crop.jpg")
        cropped_img.save(tmp_path)

        return load_image(tmp_path, resize=target_size).to(device)

    except Exception as e:
        try:
            img = Image.open(base_path / row['path']).convert('RGB')
            tmp_path = Path("tmp_crop.jpg")
            img.save(tmp_path)
            return load_image(tmp_path, resize=target_size).to(device)
        except Exception:
            print("[ERROR] Skipping")
            return None

def evaluate_reid(query_df, gallery_df, data_root, matcher, device):
    results = []
    base_path = Path(data_root)

    print(f"Evaluating Re-ID. Query: {len(query_df)} | Gallery: {len(gallery_df)}")

    for q_idx, q_row in tqdm(query_df.iterrows(), total=len(query_df), desc="Query loop"):
        q_tensor = load_and_crop_via_rle(q_row, base_path, device)
        if q_tensor is None:
            continue

        for g_idx, g_row in gallery_df.iterrows():
            if q_row['path'] == g_row['path']:
                continue

            g_tensor = load_and_crop_via_rle(g_row, base_path, device)
            if g_tensor is None:
                continue

            matches_count = matcher.get_matches_count_from_tensors(q_tensor, g_tensor, conf_threshold=0.1)
            is_same_lynx = (q_row['identity'] == g_row['identity'])

            results.append({
                'query_img': q_row['path'],
                'gallery_img': g_row['path'],
                'query_identity': q_row['identity'],
                'gallery_identity': g_row['identity'],
                'is_same_ground_truth': is_same_lynx,
                'lightglue_matches': matches_count
            })

    return pd.DataFrame(results)

def get_random_subset(df, n_per_identity=5):
    return df.groupby('identity').apply(lambda x: x.sample(n=min(len(x), n_per_identity))).reset_index(drop=True)


def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = LocalMatcher(device)
    data_module = CzechLynxDataModule(cfg=cfg)

    # train_calibrator_query_df = data_module.train_calibrator_query_ds.df
    # train_calibrator_gallery_df = data_module.train_calibrator_gallery_ds.df
    #
    # results_df = evaluate_reid(
    #     query_df=train_calibrator_query_df,
    #     gallery_df=train_calibrator_gallery_df,
    #     data_root=cfg.data_root,
    #     matcher=matcher,
    #     device=device
    # )
    #
    # output_path = "reid_lightglue_results.csv"
    # results_df.to_csv(output_path, index=False)
    # print(f"\nFile saved to: {output_path}")

    val_query_df = get_calibration_subset(data_module.val_query_ds.df, n_per_identity=4, max_total=200)
    val_gallery_df = data_module.val_gallery_ds.df

    results_df = evaluate_reid(
        query_df=val_query_df,
        gallery_df=val_gallery_df,
        data_root=cfg.data_root,
        matcher=matcher,
        device=device
    )

    output_path = "reid_lightglue_val_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nFile saved to: {output_path}")


if __name__ == "__main__":
    main()