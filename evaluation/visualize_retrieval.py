import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch
from torch.utils.data import DataLoader

from models.base import BaseModel


def _load_pil(dataset, idx: int) -> Image.Image:
    return Image.open(dataset.data_root / dataset.df.iloc[idx]["path"]).convert("RGB")


def _extract_all_embeddings(
    model: BaseModel,
    dataset,
    device: torch.device,
    val_transform,
    max_samples: int | None = None,
) -> tuple[np.ndarray, list[str], list[int]]:
    model.eval()
    embeddings, identities, indices = [], [], []

    all_idx = list(range(len(dataset)))
    if max_samples and len(all_idx) > max_samples:
        import random
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

    return np.array(embeddings), identities, indices


def run_retrieval_examples(
    model: BaseModel,
    query_dataset,
    gallery_dataset,
    device: torch.device,
    val_transform,
    n_queries: int = 5,
    top_k: int = 5,
    save_path: str = "retrieval_examples.png",
):
    """
    For n_queries query images, show top_k gallery retrievals.
    Green border = correct identity, red = wrong.
    Queries are sampled to have diverse identities.
    """
    print("Extracting query embeddings...")
    q_embs, q_ids, q_idxs = _extract_all_embeddings(model, query_dataset, device, val_transform)
    print("Extracting gallery embeddings...")
    g_embs, g_ids, g_idxs = _extract_all_embeddings(model, gallery_dataset, device, val_transform)

    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-8)
    g_norm = g_embs / (np.linalg.norm(g_embs, axis=1, keepdims=True) + 1e-8)
    sim = q_norm @ g_norm.T  # [Q, G]

    seen_ids, chosen = set(), []
    for i, qid in enumerate(q_ids):
        if qid not in seen_ids:
            chosen.append(i)
            seen_ids.add(qid)
        if len(chosen) >= n_queries:
            break

    fig, axes = plt.subplots(n_queries, top_k + 1, figsize=((top_k + 1) * 2.5, n_queries * 2.8))
    fig.suptitle("Retrieval Examples — Query (left) + Top-K Gallery Results", fontsize=13)

    for row, q_i in enumerate(chosen):
        ax = axes[row, 0]
        ax.imshow(_load_pil(query_dataset, q_idxs[q_i]).resize((160, 160)))
        ax.set_title(f"Query\n{q_ids[q_i]}", fontsize=6, color="black")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_edgecolor("blue")

        top_k_idx = np.argsort(sim[q_i])[::-1][:top_k]
        for col, g_i in enumerate(top_k_idx):
            ax = axes[row, col + 1]
            ax.imshow(_load_pil(gallery_dataset, g_idxs[g_i]).resize((160, 160)))
            correct = (g_ids[g_i] == q_ids[q_i])
            color = "green" if correct else "red"
            ax.set_title(f"{'✓' if correct else '✗'} {g_ids[g_i]}\nsim={sim[q_i][g_i]:.2f}",
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


def run_confusion_analysis(
    model: BaseModel,
    query_dataset,
    gallery_dataset,
    device: torch.device,
    val_transform,
    n_examples: int = 6,
    save_path: str = "confusion_analysis.png",
):
    """
    Show cases where model is most confidently wrong —
    high similarity score but wrong identity.
    These are the hardest confusions, great for presentation.
    """
    print("Extracting embeddings for confusion analysis...")
    q_embs, q_ids, q_idxs = _extract_all_embeddings(model, query_dataset, device, val_transform)
    g_embs, g_ids, g_idxs = _extract_all_embeddings(model, gallery_dataset, device, val_transform)

    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-8)
    g_norm = g_embs / (np.linalg.norm(g_embs, axis=1, keepdims=True) + 1e-8)
    sim = q_norm @ g_norm.T

    confusions = []
    for q_i in range(len(q_ids)):
        top1_g = np.argmax(sim[q_i])
        if g_ids[top1_g] != q_ids[q_i]:
            confusions.append((sim[q_i][top1_g], q_i, top1_g))

    confusions.sort(reverse=True)
    confusions = confusions[:n_examples]

    fig, axes = plt.subplots(n_examples, 2, figsize=(6, n_examples * 3))
    fig.suptitle("Most Confident Wrong Retrievals", fontsize=13)

    for row, (score, q_i, g_i) in enumerate(confusions):
        axes[row, 0].imshow(_load_pil(query_dataset, q_idxs[q_i]).resize((200, 200)))
        axes[row, 0].set_title(f"Query: {q_ids[q_i]}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(_load_pil(gallery_dataset, g_idxs[g_i]).resize((200, 200)))
        axes[row, 1].set_title(f"Retrieved: {g_ids[g_i]}\nsim={score:.3f}", fontsize=8, color="red")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def run_embedding_distance_distribution(
    model: BaseModel,
    query_dataset,
    gallery_dataset,
    device: torch.device,
    val_transform,
    save_path: str = "distance_distribution.png",
    max_samples: int = 1000,
):
    """
    Distribution of cosine similarities:
    - same identity (positive pairs) vs different identity (negative pairs).
    Good separation = good model.
    """
    print("Extracting embeddings for distance distribution...")
    q_embs, q_ids, _ = _extract_all_embeddings(
        model, query_dataset, device, val_transform, max_samples=max_samples
    )
    g_embs, g_ids, _ = _extract_all_embeddings(
        model, gallery_dataset, device, val_transform
    )

    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-8)
    g_norm = g_embs / (np.linalg.norm(g_embs, axis=1, keepdims=True) + 1e-8)
    sim = q_norm @ g_norm.T

    pos_sims, neg_sims = [], []
    for q_i, qid in enumerate(q_ids):
        for g_i, gid in enumerate(g_ids):
            s = sim[q_i, g_i]
            if qid == gid:
                pos_sims.append(s)
            else:
                neg_sims.append(s)

    import random
    random.seed(42)
    if len(neg_sims) > 50000:
        neg_sims = random.sample(neg_sims, 50000)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(neg_sims, bins=80, alpha=0.6, color="red", label="Different identity (negatives)", density=True)
    ax.hist(pos_sims, bins=80, alpha=0.7, color="green", label="Same identity (positives)", density=True)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Embedding Distance Distribution — {model.__class__.__name__}")
    ax.legend(fontsize=10)
    ax.axvline(x=np.mean(pos_sims), color="darkgreen", linestyle="--",
               label=f"Mean positive: {np.mean(pos_sims):.3f}")
    ax.axvline(x=np.mean(neg_sims), color="darkred", linestyle="--",
               label=f"Mean negative: {np.mean(neg_sims):.3f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")