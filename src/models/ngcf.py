"""
src/models/ngcf.py — Neural Graph Collaborative Filtering
"""

import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # ── Embedding tables ──────────────────────────────────────────────
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # ── NGCF layer transforms ─────────────────────────────────────────
        self.W1 = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=True)
            for _ in range(num_layers)
        ])

        self.W2 = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=True)
            for _ in range(num_layers)
        ])

        for l in range(num_layers):
            nn.init.xavier_uniform_(self.W1[l].weight)
            nn.init.xavier_uniform_(self.W2[l].weight)

        # ── Cached normalized adjacency ───────────────────────────────────
        self._norm_adj: torch.Tensor | None = None

        # ── Inference cache ───────────────────────────────────────────────
        self._cached_user_emb: torch.Tensor | None = None
        self._cached_item_emb: torch.Tensor | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Public Helpers
    # ──────────────────────────────────────────────────────────────────────

    def precompute_norm_adj(
        self,
        edge_index: torch.Tensor,
        n_nodes: int,
    ):
        """
        Build normalized adjacency:
            D^{-1/2} A D^{-1/2}

        Cached as sparse COO tensor.
        """

        device = edge_index.device

        row = edge_index[0]
        col = edge_index[1]

        # Degree vector
        deg = torch.zeros(
            n_nodes,
            dtype=torch.float32,
            device=device,
        )

        deg.scatter_add_(
            0,
            row,
            torch.ones(
                row.size(0),
                dtype=torch.float32,
                device=device,
            ),
        )

        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float("inf")] = 0.0

        # Edge normalization
        norm_vals = d_inv_sqrt[row] * d_inv_sqrt[col]

        self._norm_adj = torch.sparse_coo_tensor(
            edge_index,
            norm_vals,
            size=(n_nodes, n_nodes),
            dtype=torch.float32,
            device=device,
        ).coalesce()

        print(
            f"[NGCF] norm_adj built | "
            f"nodes={n_nodes:,}  edges={row.size(0):,}"
        )

    def init_item_embeddings_from_clip(
        self,
        item_feat: torch.Tensor,
    ):
        """
        Warm-start item embeddings using CLIP embeddings.
        """

        n, clip_dim = item_feat.shape

        if clip_dim != self.embedding_dim:

            proj = nn.Linear(
                clip_dim,
                self.embedding_dim,
                bias=False,
            ).to(item_feat.device)

            nn.init.xavier_uniform_(proj.weight)

            with torch.no_grad():
                projected = proj(item_feat)

        else:
            projected = item_feat

        with torch.no_grad():

            projected = projected.float()

            norms = projected.norm(
                dim=1,
                keepdim=True,
            ).clamp(min=1e-8)

            self.item_embedding.weight[:n].copy_(
                projected / norms
            )

        print(
            f"[NGCF] item_embedding initialized "
            f"from CLIP ({clip_dim}→{self.embedding_dim})"
        )

    def invalidate_cache(self):

        self._cached_user_emb = None
        self._cached_item_emb = None

    # ──────────────────────────────────────────────────────────────────────
    # Core NGCF Propagation
    # ──────────────────────────────────────────────────────────────────────

    def _propagate(self):

        assert self._norm_adj is not None, (
            "precompute_norm_adj() chưa được gọi."
        )

        # Initial embeddings
        ego = torch.cat(
            [
                self.user_embedding.weight,
                self.item_embedding.weight,
            ],
            dim=0,
        )  # (N, D)

        all_layer_embs = [ego]

        for l in range(self.num_layers):
            with amp.autocast("cuda", enabled=False):

                ego_f32 = ego.float()

                norm_adj_f32 = self._norm_adj.float()

                # Neighborhood aggregation
                neigh = torch.sparse.mm(
                    norm_adj_f32,
                    ego_f32,
                )

                # NGCF bi-interaction
                interaction = ego_f32 * neigh

            # Back to original dtype (fp16 if AMP enabled)
            neigh = neigh.to(ego.dtype)
            interaction = interaction.to(ego.dtype)


            ego = F.leaky_relu(
                self.W1[l](ego + neigh)
                + self.W2[l](interaction),
                negative_slope=0.2,
            )

            if self.dropout > 0:
                ego = F.dropout(
                    ego,
                    p=self.dropout,
                    training=self.training,
                )

            # L2 normalize
            ego = F.normalize(
                ego,
                p=2,
                dim=1,
            )

            all_layer_embs.append(ego)

        # Mean pooling across layers
        out = torch.stack(
            all_layer_embs,
            dim=0,
        ).mean(dim=0)

        user_emb = out[:self.num_users]
        item_emb = out[self.num_users:]

        return user_emb, item_emb

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        edge_index: torch.Tensor = None,
        use_cache: bool = False,
    ):
        """
        edge_index is ignored because adjacency
        is already precomputed/cached.
        """

        if (
            use_cache
            and self._cached_user_emb is not None
        ):
            return (
                self._cached_user_emb,
                self._cached_item_emb,
            )

        user_emb, item_emb = self._propagate()

        if use_cache:

            self._cached_user_emb = user_emb
            self._cached_item_emb = item_emb

        return user_emb, item_emb

    # ──────────────────────────────────────────────────────────────────────
    # Predict
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ):
        """
        Raw dot-product score.
        """

        user_emb, item_emb = self.forward(
            use_cache=True
        )

        scores = (
            user_emb[user_ids]
            * item_emb[item_ids]
        ).sum(dim=-1)

        return scores