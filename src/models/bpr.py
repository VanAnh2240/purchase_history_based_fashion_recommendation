# File src/models/bpr.py

import torch
import torch.nn as nn


class BPRMatrixFactorization(nn.Module):
    """
    BPR-MF (Bayesian Personalized Ranking)

    Baseline recommendation truyền thống (không dùng graph)

    Ý tưởng:
    - Học embedding user và item
    - Tối ưu ranking: user thích item positive hơn item negative
    """

    def __init__(self, num_users, num_items, embedding_dim=64):
        super(BPRMatrixFactorization, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # embedding cho user và item
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_ids, item_pos, item_neg):
        """
        Forward cho BPR loss training

        Args:
            user_ids: user batch
            item_pos: item user đã tương tác (positive)
            item_neg: item user chưa tương tác (negative)
        """

        user_vec = self.user_embedding(user_ids)
        pos_vec = self.item_embedding(item_pos)
        neg_vec = self.item_embedding(item_neg)

        # score positive và negative
        pos_score = torch.sum(user_vec * pos_vec, dim=1)
        neg_score = torch.sum(user_vec * neg_vec, dim=1)

        return pos_score, neg_score

    def predict(self, user_ids, item_ids):
        """
        Dự đoán score user-item (inference)
        """
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)

        return torch.sum(user_vec * item_vec, dim=1)