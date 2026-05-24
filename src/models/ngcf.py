"""
src/models/ngcf.py — Neural Graph Collaborative Filtering
"""

import torch
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
        self.num_users     = num_users
        self.num_items     = num_items
        self.embedding_dim = embedding_dim
        self.num_layers    = num_layers
        self.dropout       = dropout

        # ── Embedding tables ──────────────────────────────────────────────
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # ── Per-layer transformation matrices (W1, W2) ────────────────────
        # W1: self-interaction  (dim → dim)
        # W2: neighbour-interaction (dim → dim)
        self.W1 = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=True)
            for _ in range(num_layers)
        ])
        self.W2 = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=True)
            for _ in range(num_layers)
        ])
        for layer in range(num_layers):
            nn.init.xavier_uniform_(self.W1[layer].weight)
            nn.init.xavier_uniform_(self.W2[layer].weight)

        # ── Normalised adjacency (cached after precompute_norm_adj) ───────
        self._norm_adj: torch.Tensor | None = None

        # ── Inference cache ───────────────────────────────────────────────
        self._cached_user_emb: torch.Tensor | None = None
        self._cached_item_emb: torch.Tensor | None = None

    # ── Public helpers (expected by trainer) ──────────────────────────────

    def precompute_norm_adj(self, edge_index: torch.Tensor, n_nodes: int):
        """
        Tính D^{-1/2} A D^{-1/2} và cache lại dưới dạng sparse COO tensor.
        Gọi 1 lần trước khi train, không cần gọi lại.
        """
        device = edge_index.device

        row = edge_index[0]
        col = edge_index[1]

        # Degree vector
        deg = torch.zeros(n_nodes, dtype=torch.float32, device=device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), dtype=torch.float32, device=device))

        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float("inf")] = 0.0

        # Normalise edge weights: d_i^{-1/2} * d_j^{-1/2}
        norm_vals = d_inv_sqrt[row] * d_inv_sqrt[col]

        self._norm_adj = torch.sparse_coo_tensor(
            edge_index,
            norm_vals,
            size=(n_nodes, n_nodes),
            device=device,
        ).coalesce()

        print(f"[NGCF] norm_adj built | nodes={n_nodes:,}  edges={row.size(0):,}")

    def init_item_embeddings_from_clip(self, item_feat: torch.Tensor):
        """
        Warm-start item_embedding.weight từ CLIP/FashionCLIP embedding.
        item_feat: (n_items, clip_dim) — sẽ được project nếu clip_dim ≠ embedding_dim.
        """
        n, clip_dim = item_feat.shape
        if clip_dim != self.embedding_dim:
            # Linear projection tạm thời (không lưu weight)
            proj = nn.Linear(clip_dim, self.embedding_dim, bias=False).to(item_feat.device)
            nn.init.xavier_uniform_(proj.weight)
            with torch.no_grad():
                projected = proj(item_feat)
        else:
            projected = item_feat

        with torch.no_grad():
            # Normalize về unit sphere trước khi inject
            norms = projected.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.item_embedding.weight[:n].copy_(projected / norms)

        print(f"[NGCF] item_embedding initialized from CLIP ({clip_dim}→{self.embedding_dim})")

    def invalidate_cache(self):
        self._cached_user_emb = None
        self._cached_item_emb = None

    # ── Core NGCF propagation ─────────────────────────────────────────────

    def _propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        NGCF message passing:
          e^{(l+1)}_u = LeakyReLU( W1 * e^{(l)}_u
                                   + W1 * (A_norm @ e^{(l)})_u     [neighbourhood mean]
                                   + W2 * (A_norm @ (e^{(l)}_u ⊙ e^{(l)}))_u )  [interaction]
        Final embedding = concat of all layer outputs.
        """
        assert self._norm_adj is not None, (
            "precompute_norm_adj() chưa được gọi trước khi forward()."
        )

        device = self.user_embedding.weight.device

        # All node embeddings in one matrix: [n_users + n_items, dim]
        ego = torch.cat([self.user_embedding.weight,
                         self.item_embedding.weight], dim=0)  # (N, D)

        all_layer_embs = [ego]

        for l in range(self.num_layers):
            # Neighbour aggregation: A_norm @ e^{(l)}
            neigh = torch.sparse.mm(self._norm_adj, ego)               # (N, D)

            # Element-wise interaction: A_norm @ (e^{(l)} ⊙ neigh)
            interaction = torch.sparse.mm(self._norm_adj, ego * neigh) # (N, D)

            # NGCF update rule
            ego_new = F.leaky_relu(
                self.W1[l](ego + neigh) + self.W2[l](interaction),
                negative_slope=0.2,
            )
            ego_new = F.dropout(ego_new, p=self.dropout, training=self.training)

            ego = ego_new
            all_layer_embs.append(ego)

        # Concat all layers → richer representation
        out = torch.cat(all_layer_embs, dim=-1)           # (N, D*(L+1))

        # Split user / item
        user_emb = out[:self.num_users]
        item_emb = out[self.num_users:]
        return user_emb, item_emb

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        edge_index: torch.Tensor,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        edge_index : không được dùng trực tiếp (đã precompute trong norm_adj),
                     giữ signature để tương thích với trainer chung.
        use_cache  : nếu True và đã có cache → trả về ngay.
        """
        if use_cache and self._cached_user_emb is not None:
            return self._cached_user_emb, self._cached_item_emb

        user_emb, item_emb = self._propagate()

        if use_cache:
            self._cached_user_emb = user_emb
            self._cached_item_emb = item_emb

        return user_emb, item_emb

    # ── predict (dùng cho inference / runner.py) ──────────────────────────

    @torch.no_grad()
    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Trả về raw dot-product score (không sigmoid) cho từng cặp (user, item).
        """
        user_emb, item_emb = self.forward(None, use_cache=True)
        score = (user_emb[user_ids] * item_emb[item_ids]).sum(dim=-1)
        return score