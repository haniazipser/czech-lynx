import argparse
import os
import pandas as pd
import torch
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

from config.presets.megadesc import get_config
from data.dataloader import CzechLynxDataModule
from data.transforms import get_transforms
from models.reid import MegaDescriptorModel


def extract_embedding_for_row(row, base_path, model, transform, device):

    try:
        img_path = base_path / row['path']
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img)

        tensor = tensor.unsqueeze(0).to(device)

        model.eval()

        with torch.no_grad():
            embedding = model(tensor)
        return embedding.squeeze(0).cpu()

    except Exception as e:
        print(f"\n[ERROR] Failed to get embedding for {row['path']}: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None,
                        help="Existing wandb run ID")
    return parser.parse_args()

def main():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_id = "test"
    checkpoint_path = f"run/{run_id}/checkpoints/best.pt"

    print(f"Loading MegaDescriptor from checkpoint: {checkpoint_path}")
    model = MegaDescriptorModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    dm = CzechLynxDataModule(cfg)
    train_calibrator_query_df = dm.train_calibrator_query_ds.df
    train_calibrator_gallery_df = dm.train_calibrator_gallery_ds.df

    _, val_transform = get_transforms(cfg.experiment_type)

    query_embs = {}
    for _, row in tqdm(train_calibrator_query_df.iterrows(), total=len(train_calibrator_query_df)):
        emb = extract_embedding_for_row(row, cfg.data_root, model, val_transform, device)
        if emb is not None:
            query_embs[row['path']] = emb

    gallery_embs = {}
    for _, row in tqdm(train_calibrator_gallery_df.iterrows(), total=len(train_calibrator_gallery_df)):
        emb = extract_embedding_for_row(row, cfg.data_root, model, val_transform, device)
        if emb is not None:
            gallery_embs[row['path']] = emb

    lightglue_csv_path = "reid_lightglue_results.csv"

    if not os.path.exists(lightglue_csv_path):
        raise FileNotFoundError(f"File {lightglue_csv_path} not found")

    df_fusion = pd.read_csv(lightglue_csv_path)

    megadesc_scores = []

    for _, row in tqdm(df_fusion.iterrows(), total=len(df_fusion), desc="Connecting pairs"):
        q_path = row['query_img']
        g_path = row['gallery_img']

        if q_path in query_embs and g_path in gallery_embs:
            emb_q = query_embs[q_path]
            emb_g = gallery_embs[g_path]

            cos_sim = F.cosine_similarity(emb_q.unsqueeze(0), emb_g.unsqueeze(0)).item()
            megadesc_scores.append(cos_sim)
        else:
            megadesc_scores.append(0.0)

    df_fusion['megadesc_score'] = megadesc_scores

    output_fusion_path = "fusion_combined_scores.csv"
    df_fusion.to_csv(output_fusion_path, index=False)

    print(f"\nResult saved to: {output_fusion_path}")

if __name__ == "__main__":
    main()