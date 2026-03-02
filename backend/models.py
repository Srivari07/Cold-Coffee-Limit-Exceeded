"""
PyTorch model class definitions for CSAO backend inference.

Extracted from 04_model_training.py — same architectures used at training time.
Models loaded at startup for real-time ML-powered recommendations.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Any


# ==============================================================================
# Two-Tower Model (Candidate Generation)
# ==============================================================================

class AttentionCartPooling(nn.Module):
    """Learnable attention-weighted pooling over cart item embeddings."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_dim // 2, 1),
        )

    def forward(self, cart_emb: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.attention[0](cart_emb))
        gate = self.attention[2](torch.tanh(gate))
        return cart_emb * torch.sigmoid(gate)


class QueryTower(nn.Module):
    """Query tower: user features + attention-pooled cart + sequential context + context -> hidden_dim."""

    def __init__(self, user_dim: int, cart_dim: int, context_dim: int,
                 hidden_dim: int = 64, seq_dim: int = 0) -> None:
        super().__init__()
        self.seq_dim = seq_dim
        self.cart_attention = AttentionCartPooling(cart_dim)
        total_input = user_dim + cart_dim + context_dim + seq_dim
        self.network = nn.Sequential(
            nn.Linear(total_input, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, user_feats: torch.Tensor, cart_feats: torch.Tensor,
                context_feats: torch.Tensor,
                seq_feats: torch.Tensor | None = None) -> torch.Tensor:
        cart_pooled = self.cart_attention(cart_feats)
        parts = [user_feats, cart_pooled, context_feats]
        if seq_feats is not None and self.seq_dim > 0:
            parts.append(seq_feats)
        combined = torch.cat(parts, dim=1)
        return self.network(combined)


class ItemTower(nn.Module):
    """Item tower: item features -> hidden_dim."""

    def __init__(self, item_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(item_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, hidden_dim),
        )

    def forward(self, item_feats: torch.Tensor) -> torch.Tensor:
        return self.network(item_feats)


class TwoTowerModel(nn.Module):
    """Two-Tower retrieval model: score = dot(query_emb, item_emb)."""

    def __init__(self, user_dim: int, cart_dim: int, context_dim: int,
                 item_dim: int, hidden_dim: int = 64, seq_dim: int = 0) -> None:
        super().__init__()
        self.query_tower = QueryTower(user_dim, cart_dim, context_dim, hidden_dim, seq_dim)
        self.item_tower = ItemTower(item_dim, hidden_dim)

    def forward(self, user_feats: torch.Tensor, cart_feats: torch.Tensor,
                context_feats: torch.Tensor, item_feats: torch.Tensor,
                seq_feats: torch.Tensor | None = None):
        query_emb = self.query_tower(user_feats, cart_feats, context_feats, seq_feats)
        item_emb = self.item_tower(item_feats)
        query_emb = nn.functional.normalize(query_emb, p=2, dim=1)
        item_emb = nn.functional.normalize(item_emb, p=2, dim=1)
        score = (query_emb * item_emb).sum(dim=1)
        return score, query_emb, item_emb


# ==============================================================================
# SASRec Sequential Model
# ==============================================================================

class SASRecModel(nn.Module):
    """Self-Attentive Sequential Recommendation model (SASRec).

    2-layer Transformer encoder processes cart item sequences.
    Produces a sequential context embedding for downstream ranking.
    """

    def __init__(self, num_items: int, emb_dim: int = 64, n_heads: int = 2,
                 n_layers: int = 2, max_seq_len: int = 20, dropout: float = 0.1) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len

        self.item_embedding = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.output_proj = nn.Linear(emb_dim, num_items + 1)

    def forward(self, item_seq: torch.Tensor):
        batch_size, seq_len = item_seq.shape
        seq_len = min(seq_len, self.max_seq_len)
        item_seq = item_seq[:, :seq_len]

        item_emb = self.item_embedding(item_seq)
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)

        x = self.dropout(item_emb + pos_emb)
        x = self.layer_norm(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_seq.device), diagonal=1
        ).bool()
        padding_mask = (item_seq == 0)

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.output_proj(x)

        lengths = (item_seq != 0).sum(dim=1).clamp(min=1)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.emb_dim)
        context_emb = x.gather(1, idx).squeeze(1)

        return logits, context_emb


# ==============================================================================
# DCN-v2 (Deep & Cross Network v2)
# ==============================================================================

class CrossNetwork(nn.Module):
    """DCN-v2 Cross Network: learned cross layers for explicit feature interactions."""

    def __init__(self, input_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
            for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for i in range(self.num_layers):
            xw = torch.matmul(x, self.W[i]) + self.b[i]
            x = x0 * xw + x
        return x


class DeepNetwork(nn.Module):
    """DCN-v2 Deep Network: MLP with ReLU + dropout."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DCNv2Model(nn.Module):
    """DCN-v2: Cross Network + Deep Network -> linear -> logit."""

    def __init__(self, input_dim: int, num_cross_layers: int = 3,
                 deep_dims: list[int] | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        if deep_dims is None:
            deep_dims = [256, 128, 64]

        self.cross_network = CrossNetwork(input_dim, num_cross_layers)
        self.deep_network = DeepNetwork(input_dim, deep_dims, dropout)
        self.final_linear = nn.Linear(input_dim + deep_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)
        combined = torch.cat([cross_out, deep_out], dim=1)
        logit = self.final_linear(combined).squeeze(1)
        return logit


# ==============================================================================
# MMR Diversity Re-ranking
# ==============================================================================

def mmr_rerank(
    candidates: list[dict[str, Any]],
    embeddings: np.ndarray | None = None,
    lambda_param: float = 0.7,
    top_k: int = 8,
    max_per_category: int = 2,
    freq_cap: dict[str, int] | None = None,
    impression_penalty: dict[str, float] | None = None,
    prev_session_items: set[str] | None = None,
) -> list[dict[str, Any]]:
    """MMR re-ranking with category diversity and fatigue prevention.

    At each step, pick item maximizing:
        lambda * relevance - (1-lambda) * max_similarity_to_selected

    Enforces: max 2 items per category, at least 2 different categories.

    Fatigue prevention:
    - Frequency capping: skip items shown >= 3 times today
    - Impression decay: -0.05 per show-without-click reduces score
    - Session novelty: at least 2 of top 5 must be new vs previous request
    """
    if len(candidates) == 0:
        return candidates

    scores = np.array([c["score"] for c in candidates], dtype=np.float64)

    # Normalize scores to [0, 1]
    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        norm_scores = (scores - score_min) / (score_max - score_min)
    else:
        norm_scores = np.ones(len(scores)) * 0.5

    # Fatigue: impression decay penalty (applied before selection)
    if impression_penalty:
        for i, c in enumerate(candidates):
            penalty = impression_penalty.get(c["item_id"], 0.0)
            if penalty > 0:
                norm_scores[i] = max(0.0, norm_scores[i] - penalty)

    # Build similarity matrix from embeddings if available
    if embeddings is not None and len(embeddings) > 0:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        norm_emb = embeddings / norms
        sim_matrix = norm_emb @ norm_emb.T
    else:
        sim_matrix = np.zeros((len(scores), len(scores)))

    selected_indices: list[int] = []
    remaining = list(range(len(scores)))
    category_counts: dict[str, int] = defaultdict(int)
    novel_in_top5 = 0

    for step in range(min(top_k, len(candidates))):
        best_score = -float("inf")
        best_idx = None

        for idx in remaining:
            # Fatigue: frequency capping (skip items shown >= 3 times today)
            if freq_cap and freq_cap.get(candidates[idx]["item_id"], 0) >= 3:
                continue

            relevance = norm_scores[idx]

            if selected_indices:
                max_sim = max(sim_matrix[idx][s] for s in selected_indices)
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            cat = candidates[idx].get("category", "unknown")
            if category_counts[cat] >= max_per_category:
                mmr_score -= 0.5

            # Fatigue: session novelty boost (for top-5 positions only)
            if step < 5 and prev_session_items is not None:
                if candidates[idx]["item_id"] not in prev_session_items:
                    if novel_in_top5 < 2:
                        mmr_score += 0.1

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        cat = candidates[best_idx].get("category", "unknown")
        category_counts[cat] += 1

        # Track novelty for top 5
        if step < 5 and prev_session_items is not None:
            if candidates[best_idx]["item_id"] not in prev_session_items:
                novel_in_top5 += 1

    # Fatigue: enforce min 2 novel items in top 5 (swap if needed)
    if prev_session_items is not None and len(selected_indices) >= 5 and novel_in_top5 < 2:
        for swap_pos in [4, 3]:
            if novel_in_top5 >= 2:
                break
            swap_id = candidates[selected_indices[swap_pos]]["item_id"]
            if swap_id in prev_session_items:
                for r_idx in remaining:
                    r_id = candidates[r_idx]["item_id"]
                    if r_id not in prev_session_items:
                        if not freq_cap or freq_cap.get(r_id, 0) < 3:
                            old_idx = selected_indices[swap_pos]
                            selected_indices[swap_pos] = r_idx
                            remaining.remove(r_idx)
                            remaining.append(old_idx)
                            novel_in_top5 += 1
                            break

    # Ensure at least 2 categories
    selected_cats = set(candidates[i].get("category", "unknown") for i in selected_indices)
    if len(selected_cats) < 2 and remaining:
        for idx in remaining:
            cat = candidates[idx].get("category", "unknown")
            if cat not in selected_cats:
                if not freq_cap or freq_cap.get(candidates[idx]["item_id"], 0) < 3:
                    selected_indices.append(idx)
                    remaining.remove(idx)
                    selected_cats.add(cat)
                    if len(selected_cats) >= 2:
                        break

    return [candidates[i] for i in selected_indices]
