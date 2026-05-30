import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.manifold import TSNE

from models.base import BaseModel

def run_tsne_analysis(
    model: BaseModel,
    dataset,
    device: torch.device,
    val_transform,
    color_by: list[str] = ["identity", "source"],
    save_path: str = "tsne_analysis.png",
    max_samples: int = 2000,
):
    model.eval()
    embeddings, meta = [], {col: [] for col in color_by}

    indices = list(range(len(dataset)))
    if len(indices) > max_samples:
        import random
        random.seed(42)
        indices = random.sample(indices, max_samples)

    with torch.no_grad():
        for idx in indices:
            row = dataset.df.iloc[idx]
            img_pil = Image.open(dataset.data_root / row["path"]).convert("RGB")
            img_tensor = val_transform(img_pil).unsqueeze(0).to(device)
            emb = model.encode(img_tensor).cpu().numpy()
            embeddings.append(emb[0])
            for col in color_by:
                meta[col].append(row.get(col, "unknown"))

    embeddings = np.array(embeddings)

    coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)

    n_plots = len(color_by)
    fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(f"t-SNE of {model.__class__.__name__} Embeddings", fontsize=14)

    for ax, col in zip(axes, color_by):
        values  = meta[col]
        unique  = list(set(values))
        val_map = {v: i for i, v in enumerate(unique)}
        colors  = [val_map[v] for v in values]

        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                             c=colors, cmap="tab20", s=4, alpha=0.6)
        ax.set_title(f"Colored by: {col}", fontsize=11)
        ax.axis("off")

        if len(unique) <= 20:
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=plt.cm.tab20(val_map[v] / max(len(unique)-1, 1)),
                                      label=str(v)) for v in unique]
            ax.legend(handles=patches, loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def run_tsne_camera_vs_identity(
    model: BaseModel,
    dataset,
    device: torch.device,
    val_transform,
    focus_camera: str = None,   # np. "foe_carpaths-341"
    save_path: str = "tsne_camera_identity.png",
    max_samples: int = 2000,
):
    model.eval()
    embeddings, identities, trap_ids = [], [], []

    indices = list(range(len(dataset)))
    if len(indices) > max_samples:
        import random
        random.seed(42)
        indices = random.sample(indices, max_samples)

    with torch.no_grad():
        for idx in indices:
            row = dataset.df.iloc[idx]
            img_pil = Image.open(dataset.data_root / row["path"]).convert("RGB")
            img_tensor = val_transform(img_pil).unsqueeze(0).to(device)
            emb = model.encode(img_tensor).cpu().numpy()
            embeddings.append(emb[0])
            identities.append(row["identity"])
            trap_ids.append(row["trap_id"])

    embeddings = np.array(embeddings)
    coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)

    if focus_camera is None:
        from collections import Counter, defaultdict
        camera_identities = defaultdict(set)
        for trap, ident in zip(trap_ids, identities):
            camera_identities[trap].add(ident)
        focus_camera = max(camera_identities, key=lambda c: len(camera_identities[c]))
        print(f"Focus camera: {focus_camera} ({len(camera_identities[focus_camera])} unique animals)")

    focus_identities = list(set(
        ident for trap, ident in zip(trap_ids, identities) if trap == focus_camera
    ))
    cmap = plt.cm.tab20
    identity_to_color = {ident: cmap(i / max(len(focus_identities)-1, 1))
                         for i, ident in enumerate(focus_identities)}

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle(
        f"Shortcut Learning: Different lynxes from same camera cluster together\n"
        f"Camera: {focus_camera} ({len(focus_identities)} different animals)",
        fontsize=12
    )

    other_mask = [i for i, trap in enumerate(trap_ids) if trap != focus_camera]
    ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
               c="lightgray", s=4, alpha=0.2, zorder=1, label="other cameras")

    for ident in focus_identities:
        mask = [i for i, (trap, id_) in enumerate(zip(trap_ids, identities))
                if trap == focus_camera and id_ == ident]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[identity_to_color[ident]], s=20, alpha=0.9,
                   zorder=2, label=ident)

    ax.legend(fontsize=7, loc="upper right", ncol=2,
              title="Different lynxes, same camera")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")