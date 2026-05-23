# File src/models/ngcf.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    """
    NGCF: Neural Graph Collaborative Filtering

    Khác LightGCN:
    - Có transformation matrix (W1, W2)
    - Có nonlinear activation
    - Learning interaction phức tạp hơn

    FIX:
    - Thêm support init item embedding từ CLIP
    """

    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(NGCF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.W1 = nn.ModuleList()
        self.W2 = nn.ModuleList()

        for _ in range(num_layers):
            self.W1.append(nn.Linear(embedding_dim, embedding_dim))
            self.W2.append(nn.Linear(embedding_dim, embedding_dim))

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def init_item_embeddings_from_clip(self, item_feat: torch.Tensor):
        feat_dim = item_feat.shape[1]
        embedding_dim = self.embedding_dim

        if feat_dim != embedding_dim:
            proj = nn.Linear(feat_dim, embedding_dim, bias=False)
            nn.init.xavier_uniform_(proj.weight)
            with torch.no_grad():
                projected = proj(item_feat.float())
        else:
            projected = item_feat.float()

        with torch.no_grad():
            self.item_embedding.weight.copy_(projected)

        print(f"[NGCF] item_embedding initialized from CLIP features ✓")

    def forward(self, edge_index):
        users = self.user_embedding.weight
        items = self.item_embedding.weight

        all_emb = torch.cat([users, items], dim=0)
        layer_embs = [all_emb]

        for layer in range(self.num_layers):
            all_emb = self.propagate(edge_index, all_emb, layer)
            layer_embs.append(all_emb)

        final_emb = torch.mean(torch.stack(layer_embs, dim=0), dim=0)

        users_final, items_final = torch.split(
            final_emb,
            [self.num_users, self.num_items],
        )

        return users_final, items_final

    def propagate(self, edge_index, emb, layer):
        row, col = edge_index
        n = emb.size(0)

        # FIX: degree normalize đúng cho cả row và col (giống GraphSAGE fix)
        neigh_emb = torch.zeros_like(emb)
        neigh_emb.index_add_(0, row, emb[col])
        neigh_emb.index_add_(0, col, emb[row])

        deg_row = torch.bincount(row, minlength=n).float()
        deg_col = torch.bincount(col, minlength=n).float()
        deg = (deg_row + deg_col).unsqueeze(1).clamp(min=1e-10)
        neigh_emb = neigh_emb / deg

        self_emb    = self.W1[layer](emb)
        neigh_emb   = self.W2[layer](neigh_emb)
        interaction = self_emb * neigh_emb

        out = F.relu(self_emb + neigh_emb + interaction)

        return out

    def predict(self, users_emb, items_emb, u, i):
        return torch.sum(users_emb[u] * items_emb[i], dim=1)