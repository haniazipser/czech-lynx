import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from models.base import BaseModel


def run_tsne_analysis(
    model: BaseModel,
    dataset,
    device: torch.device,
    val_transform,
    save_path: str = "tsne_analysis.png",
    max_samples: int = 2000,
):
    model.eval()
    embeddings, identities, regions = [], [], []

    # random subset
    indices = list(range(len(dataset)))
    if len(indices) > max_samples:
        import random
        random.seed(42)
        indices = random.sample(indices, max_samples)

    with torch.no_grad():
        for idx in indices:
            img_pil = Image.open(dataset.data_root / dataset.df.iloc[idx]["path"]).convert("RGB")
            img_tensor = val_transform(img_pil).unsqueeze(0).to(device)
            emb = model.encode(img_tensor).cpu().numpy()
            embeddings.append(emb[0])
            identities.append(dataset.df.iloc[idx]["identity"])
            regions.append(dataset.df.iloc[idx]["source"])

    embeddings = np.array(embeddings)

    print("Computing t-SNE...")
    from sklearn.manifold import TSNE
    coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"t-SNE of {model.__class__.__name__} Embeddings", fontsize=14)

    unique_ids = list(set(identities))
    id_to_int  = {uid: i for i, uid in enumerate(unique_ids)}
    colors_id  = [id_to_int[l] for l in identities]
    axes[0].scatter(coords[:, 0], coords[:, 1], c=colors_id, cmap="tab20", s=4, alpha=0.6)
    axes[0].set_title("Colored by Identity\n(good model → tight clusters per animal)")
    axes[0].axis("off")

    region_colors = {"foe_carpaths": "#E24B4A", "foe_bohemia": "#1D9E75", "snpa": "#534AB7"}
    colors_reg    = [region_colors.get(r, "gray") for r in regions]
    axes[1].scatter(coords[:, 0], coords[:, 1], c=colors_reg, s=4, alpha=0.6)

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=r) for r, c in region_colors.items()]
    axes[1].legend(handles=patches, loc="upper right", fontsize=9)
    axes[1].set_title("Colored by Region\n(clusters by region → shortcut learning!)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")