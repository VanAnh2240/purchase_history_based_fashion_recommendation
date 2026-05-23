# File src/models/lightgcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    LightGCN: mô hình GNN đơn giản cho recommendation

    FIX:
    - Sửa bug forward() lồng nhau (cache không hoạt động)
    - Thêm support khởi tạo từ CLIP embedding
    - precompute_norm_adj yêu cầu edge_index đã có offset đúng
    """

    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # cache
        self._cached_user_emb = None
        self._cached_item_emb = None
        self._cache_valid = False

        # precomputed normalized adjacency (sparse)
        self._norm_adj = None

    # ── cache helpers ─────────────────────────────────────────────

    def invalidate_cache(self):
        self._cache_valid = False
        self._cached_user_emb = None
        self._cached_item_emb = None

    def init_item_embeddings_from_clip(self, item_feat: torch.Tensor):
        feat_dim = item_feat.shape[1]
    
        if feat_dim != self.embedding_dim:
            proj = nn.Linear(feat_dim, self.embedding_dim, bias=False)
            nn.init.xavier_uniform_(proj.weight)
            proj = proj.to(item_feat.device)  # ← thêm dòng này
            with torch.no_grad():
                projected = proj(item_feat.float())
            print(f"[LightGCN] CLIP feat projected: {feat_dim} → {self.embedding_dim}")
        else:
            projected = item_feat.float()
    
        with torch.no_grad():
            self.item_embedding.weight.copy_(projected)
    
        print(f"[LightGCN] item_embedding initialized from CLIP features ✓")

    # ── norm adj ──────────────────────────────────────────────────

    def precompute_norm_adj(self, edge_index: torch.Tensor, n: int):
        """
        Tính normalized adjacency matrix dạng sparse.

        Args:
            edge_index: [2, num_edges] — phải có item offset = n_users
            n: tổng số nodes = n_users + n_items
        """
        device = edge_index.device
        row, col = edge_index[0], edge_index[1]

        # degree của mỗi node (cả hai chiều vì undirected)
        deg = torch.zeros(n, dtype=torch.float32, device=device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        # col đã được đếm ở chiều ngược (graph undirected nên đã có cả chiều)

        deg_inv_sqrt = (deg + 1e-10).pow(-0.5)
        values = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        self._norm_adj = torch.sparse_coo_tensor(
            edge_index,
            values,
            size=(n, n),
            dtype=torch.float32,
            device=device,
        ).coalesce()

        print(f"[LightGCN] norm_adj precomputed: {n}×{n}, nnz={values.shape[0]:,}")

    # ── forward ───────────────────────────────────────────────────

    def forward(self, edge_index: torch.Tensor, use_cache: bool = False):
        """
        FIX: forward() không còn lồng nhau → cache hoạt động đúng.

        Args:
            edge_index: [2, num_edges] với item offset = n_users
            use_cache:  True → trả về cached embedding nếu có

        Returns:
            users_final: [n_users, embedding_dim]
            items_final: [n_items, embedding_dim]
        """
        # FIX: cache check ở đúng chỗ (không phải trong nested function)
        if use_cache and self._cache_valid:
            return self._cached_user_emb, self._cached_item_emb

        n = self.num_users + self.num_items

        if self._norm_adj is None:
            self.precompute_norm_adj(edge_index, n)

        all_emb = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight,
        ], dim=0).float()

        embs = [all_emb]

        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(self._norm_adj.float(), all_emb)
            embs.append(all_emb)

        # mean pooling qua các layer (LightGCN đặc trưng)
        final_emb = torch.stack(embs).mean(dim=0)

        users_final, items_final = torch.split(
            final_emb,
            [self.num_users, self.num_items],
        )

        if use_cache:
            self._cached_user_emb = users_final
            self._cached_item_emb = items_final
            self._cache_valid = True

        return users_final, items_final

    # ── predict / get_embedding ───────────────────────────────────

    def predict(self, users_emb, items_emb, user_indices, item_indices):
        return (users_emb[user_indices] * items_emb[item_indices]).sum(dim=1)

    def get_embedding(self, edge_index):
        return self.forward(edge_index)