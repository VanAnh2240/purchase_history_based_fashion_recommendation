# File src/models/graphsage.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    """
    GraphSAGE: mô hình GNN có neighbor aggregation

    FIX:
    - aggregate() tính degree đúng cho cả row lẫn col
      (trước đây chỉ đếm row → node chỉ xuất hiện ở col bị normalize sai)
    - Thêm support init item embedding từ CLIP
    """

    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=64):
        super(GraphSAGE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # transform sau aggregate
        self.linear = nn.Linear(embedding_dim * 2, hidden_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def init_item_embeddings_from_clip(self, item_feat: torch.Tensor):
        feat_dim = item_feat.shape[1]
        embedding_dim = self.item_embedding.embedding_dim

        if feat_dim != embedding_dim:
            proj = nn.Linear(feat_dim, embedding_dim, bias=False)
            nn.init.xavier_uniform_(proj.weight)
            with torch.no_grad():
                projected = proj(item_feat.float())
        else:
            projected = item_feat.float()

        with torch.no_grad():
            self.item_embedding.weight.copy_(projected)

        print(f"[GraphSAGE] item_embedding initialized from CLIP features ✓")

    def forward(self, edge_index):
        users = self.user_embedding.weight
        items = self.item_embedding.weight

        all_emb = torch.cat([users, items], dim=0)

        agg_emb = self.aggregate(edge_index, all_emb)

        # concat self + neighbor
        out = torch.cat([all_emb, agg_emb], dim=1)
        out = self.linear(out)
        out = F.relu(out)

        users_out, items_out = torch.split(out, [self.num_users, self.num_items])

        return users_out, items_out

    def aggregate(self, edge_index, emb):
        """
        FIX: degree tính cho cả row VÀ col.

        Trước đây:
            deg = bincount(row) → node chỉ ở col có deg≈0 → chia 1e-10 → NaN/explosion

        Sau fix:
            deg = bincount(row) + bincount(col) → đúng cho undirected graph
        """
        row, col = edge_index
        n = emb.size(0)

        out = torch.zeros_like(emb)
        out.index_add_(0, row, emb[col])
        out.index_add_(0, col, emb[row])

        # FIX: degree phải đếm cả hai chiều
        deg_row = torch.bincount(row, minlength=n).float()
        deg_col = torch.bincount(col, minlength=n).float()
        deg = (deg_row + deg_col).unsqueeze(1).clamp(min=1e-10)

        out = out / deg

        return out

    def predict(self, users_emb, items_emb, u, i):
        return torch.sum(users_emb[u] * items_emb[i], dim=1)