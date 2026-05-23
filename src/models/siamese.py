"""
src/models/siamese.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseProjector(nn.Module):
    def __init__(self, dim_in: int, dim_out: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)