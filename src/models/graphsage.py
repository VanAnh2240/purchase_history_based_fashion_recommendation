# File: src/models/graphsage.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=64, dropout=0.1):
        super().__init__()

        self.num_users     = num_users
        self.num_items     = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.linear  = nn.Linear(embedding_dim * 2, hidden_dim)
        # FIX: dropout để regularize, giảm overfit sau CLIP init
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self._adj   = None   # adjacency matrix — persist across epochs
        self._cache = None   # embedding output cache — reset each epoch

    # ───────────────────────── cache ─────────────────────────

    def invalidate_cache(self):
        self._cache = None

    # ───────────────────────── CLIP init ─────────────────────

    def init_item_embeddings_from_clip(self, item_feat: torch.Tensor):
        feat_dim = item_feat.shape[1]
        emb_dim  = self.item_embedding.embedding_dim

        if feat_dim != emb_dim:
            proj = nn.Linear(feat_dim, emb_dim, bias=False).to(item_feat.device)
            nn.init.xavier_uniform_(proj.weight)
            with torch.no_grad():
                projected = proj(item_feat.float())
        else:
            projected = item_feat.float()

        with torch.no_grad():
            self.item_embedding.weight.copy_(projected)

        print("[GraphSAGE] item_embedding initialized from CLIP features ✓")

    # ───────────────────────── adjacency build ───────────────

    def _build_adj(self, edge_index, n_nodes):
        row, col = edge_index

        indices = torch.stack([row, col], dim=0)
        values  = torch.ones(row.size(0), device=row.device)

        adj = torch.sparse_coo_tensor(
            indices,
            values,
            (n_nodes, n_nodes),
            device=row.device
        ).coalesce()

        deg     = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1e-10)
        deg_inv = 1.0 / deg
        values  = adj.values() * deg_inv[row]

        adj_norm = torch.sparse_coo_tensor(
            adj.indices(),
            values,
            adj.shape,
            device=row.device
        ).coalesce()

        return adj_norm

    # ───────────────────────── forward ───────────────────────

    def forward(self, edge_index: torch.Tensor, use_cache: bool = False):
        if use_cache and self._cache is not None:
            return self._cache

        all_emb = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)

        if self._adj is None:
            self._adj = self._build_adj(edge_index, self.num_users + self.num_items)

        agg_emb = torch.sparse.mm(self._adj, all_emb)

        out = torch.cat([all_emb, agg_emb], dim=1)
        out = self.linear(out)
        out = F.relu(out)
        out = self.dropout(out)   # FIX: dropout sau activation

        users_out, items_out = torch.split(
            out,
            [self.num_users, self.num_items]
        )

        self._cache = (users_out, items_out)
        return users_out, items_out

    # ───────────────────────── predict ───────────────────────

    def predict(self, users_emb, items_emb, u, i):
        return torch.sum(users_emb[u] * items_emb[i], dim=1)