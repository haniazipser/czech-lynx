import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from models.base import BaseModel


def _load_pil(dataset, idx: int) -> Image.Image:
    return Image.open(dataset.data_root / dataset.df.iloc[idx]["path"]).convert("RGB")


@dataclass
class EmbeddingCache:
    embeddings: np.ndarray  # [N, D] L2-normalized
    identities: list[str]
    indices: list[int]       # original dataset indices


def extract_embeddings(
    model: BaseModel,
    dataset,
    device: torch.device,
    val_transform,
    max_samples: int | None = None,
) -> EmbeddingCache:
    model.eval()
    embeddings, identities, indices = [], [], []

    all_idx = list(range(len(dataset)))
    if max_samples and len(all_idx) > max_samples:
        random.seed(42)
        all_idx = random.sample(all_idx, max_samples)

    with torch.no_grad():
        for idx in all_idx:
            row = dataset.df.iloc[idx]
            img = val_transform(_load_pil(dataset, idx)).unsqueeze(0).to(device)
            emb = model.encode(img).cpu().numpy()[0]
            embeddings.append(emb)
            identities.append(row["identity"])
            indices.append(idx)

    embs = np.array(embeddings)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    return EmbeddingCache(embs, identities, indices)


def run_all_visualizations(
    model: BaseModel,
    query_dataset,
    gallery_dataset,
    device: torch.device,
    val_transform,
    save_dir: str = ".",
    n_queries: int = 5,
    top_k: int = 5,
    n_confusions: int = 6,
    max_samples_dist: int = 1000,
):
    """Extract embeddings once, run all visualizations."""
    print("Extracting query embeddings...")
    q_cache = extract_embeddings(model, query_dataset, device, val_transform, max_samples=max_samples_dist)
    print("Extracting gallery embeddings...")
    g_cache = extract_embeddings(model, gallery_dataset, device, val_transform)

    sim = q_cache.embeddings @ g_cache.embeddings.T  # [Q, G]

    _plot_retrieval_examples(q_cache, g_cache, sim, query_dataset, gallery_dataset,
                             n_queries, top_k, f"{save_dir}/retrieval_examples.png")
    _plot_confusion(q_cache, g_cache, sim, query_dataset, gallery_dataset,
                    n_confusions, f"{save_dir}/confusion.png")
    _plot_distance_distribution(q_cache, g_cache, sim, model.__class__.__name__,
                                f"{save_dir}/distance_dist.png")


def _plot_retrieval_examples(q, g, sim, q_ds, g_ds, n_queries, top_k, save_path):
    seen_ids, chosen = set(), []
    for i, qid in enumerate(q.identities):
        if qid not in seen_ids:
            chosen.append(i)
            seen_ids.add(qid)
        if len(chosen) >= n_queries:
            break

    fig, axes = plt.subplots(n_queries, top_k + 1, figsize=((top_k + 1) * 2.5, n_queries * 2.8))
    fig.suptitle("Retrieval Examples — Query (left) + Top-K Gallery Results", fontsize=13)

    for row, q_i in enumerate(chosen):
        ax = axes[row, 0]
        ax.imshow(_load_pil(q_ds, q.indices[q_i]).resize((160, 160)))
        ax.set_title(f"Query\n{q.identities[q_i]}", fontsize=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_edgecolor("blue")

        for col, g_i in enumerate(np.argsort(sim[q_i])[::-1][:top_k]):
            ax = axes[row, col + 1]
            ax.imshow(_load_pil(g_ds, g.indices[g_i]).resize((160, 160)))
            correct = g.identities[g_i] == q.identities[q_i]
            color = "green" if correct else "red"
            ax.set_title(f"{'✓' if correct else '✗'} {g.identities[g_i]}\nsim={sim[q_i][g_i]:.2f}",
                         fontsize=6, color=color)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3)
                spine.set_edgecolor(color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def _plot_confusion(q, g, sim, q_ds, g_ds, n_examples, save_path):
    confusions = []
    for q_i in range(len(q.identities)):
        top1 = int(np.argmax(sim[q_i]))
        if g.identities[top1] != q.identities[q_i]:
            confusions.append((sim[q_i][top1], q_i, top1))

    confusions.sort(reverse=True)
    confusions = confusions[:n_examples]

    fig, axes = plt.subplots(n_examples, 2, figsize=(6, n_examples * 3))
    fig.suptitle("Most Confident Wrong Retrievals", fontsize=13)

    for row, (score, q_i, g_i) in enumerate(confusions):
        axes[row, 0].imshow(_load_pil(q_ds, q.indices[q_i]).resize((200, 200)))
        axes[row, 0].set_title(f"Query: {q.identities[q_i]}", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(_load_pil(g_ds, g.indices[g_i]).resize((200, 200)))
        axes[row, 1].set_title(f"Retrieved: {g.identities[g_i]}\nsim={score:.3f}", fontsize=8, color="red")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def _plot_distance_distribution(q, g, sim, model_name, save_path):
    pos_sims, neg_sims = [], []
    for q_i, qid in enumerate(q.identities):
        for g_i, gid in enumerate(g.identities):
            s = sim[q_i, g_i]
            (pos_sims if qid == gid else neg_sims).append(s)

    if len(neg_sims) > 50000:
        random.seed(42)
        neg_sims = random.sample(neg_sims, 50000)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(neg_sims, bins=80, alpha=0.6, color="red", label="Different identity", density=True)
    ax.hist(pos_sims, bins=80, alpha=0.7, color="green", label="Same identity", density=True)
    ax.axvline(np.mean(pos_sims), color="darkgreen", linestyle="--",
               label=f"Mean positive: {np.mean(pos_sims):.3f}")
    ax.axvline(np.mean(neg_sims), color="darkred", linestyle="--",
               label=f"Mean negative: {np.mean(neg_sims):.3f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Embedding Distance Distribution — {model_name}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")