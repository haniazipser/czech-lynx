from pathlib import Path

import torch
from torch.utils.data import DataLoader
from typing import Tuple

# ---------------------------------------------------------------------------
# Retrieval Evaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    def __init__(self, device: torch.device):
        self.device = device

        self.last_q_embeddings: torch.Tensor | None = None
        self.last_g_embeddings: torch.Tensor | None = None
        self.last_q_labels: torch.Tensor | None = None
        self.last_g_labels: torch.Tensor | None = None

    def evaluate(
            self,
            model,
            query_loader: DataLoader,
            gallery_loader: DataLoader,
    ) -> dict:
        q_emb, q_labels = self._extract_embeddings(model, query_loader)
        g_emb, g_labels = self._extract_embeddings(model, gallery_loader)

        self.last_q_embeddings = q_emb
        self.last_g_embeddings = g_emb
        self.last_q_labels = q_labels
        self.last_g_labels = g_labels

        sim_matrix = self._cosine_similarity_matrix(q_emb, g_emb)
        return self._compute_metrics(sim_matrix, q_labels, g_labels)

    def get_top_k_results(self, k: int = 5) -> dict:
        """
        Returns top-k results for each query — for retrieval examples visualization.
        Requires evaluate() to be called first.

        Returns:
            dict with keys:
                sorted_indices [n_query, k]  — indices in gallery
                sorted_labels  [n_query, k]  — gallery labels
                correct        [n_query, k]  — bool whether hit
        """
        assert self.last_q_embeddings is not None, \
            "Call evaluate first"

        sim = self._cosine_similarity_matrix(
            self.last_q_embeddings, self.last_g_embeddings
        )
        sorted_idx = sim.argsort(dim=1, descending=True)[:, :k]
        sorted_labels = self.last_g_labels[sorted_idx]
        correct = (sorted_labels == self.last_q_labels.unsqueeze(1))

        return {
            "sorted_indices": sorted_idx,
            "sorted_labels": sorted_labels,
            "correct": correct,
        }


    @torch.no_grad()
    def _extract_embeddings(
            self,
            model,
            loader: DataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        model.to(self.device)

        embeddings, labels = [], []
        for images, lbls in loader:
            images = images.to(self.device)
            emb = model.encode(images)
            embeddings.append(emb.cpu())
            labels.append(lbls.cpu())

        return torch.cat(embeddings), torch.cat(labels)

    @staticmethod
    def _cosine_similarity_matrix(
            q_emb: torch.Tensor,
            g_emb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mm(q_emb, g_emb.t())  # [Q, G]

    @staticmethod
    def _compute_metrics(
            sim_matrix: torch.Tensor,
            q_labels: torch.Tensor,
            g_labels: torch.Tensor,
            max_rank: int = 20,
    ) -> dict:
        n_query = sim_matrix.shape[0]
        max_rank = min(max_rank, sim_matrix.shape[1])

        cmc = torch.zeros(max_rank)
        ap_sum = 0.0

        for i in range(n_query):
            sorted_idx = sim_matrix[i].argsort(descending=True)
            sorted_labels = g_labels[sorted_idx]
            correct = (sorted_labels == q_labels[i]).float()

            # CMC -first hit on k position
            for k in range(max_rank):
                if correct[k] == 1.0:
                    cmc[k:] += 1.0
                    break

            # Average Precision
            n_relevant = correct.sum().item()
            if n_relevant == 0:
                continue
            precision_at_k = correct.cumsum(0) / torch.arange(1, len(correct) + 1).float()
            ap_sum += (precision_at_k * correct).sum().item() / n_relevant

        cmc = cmc / n_query

        return {
            "rank1": cmc[0].item(),
            "rank5": cmc[4].item() if max_rank >= 5 else None,
            "map": ap_sum / n_query,
            "cmc": cmc.tolist(),  # full curve
        }
