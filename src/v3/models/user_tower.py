"""
V3 用户塔。

MicroLens-100K 没有 KuaiRec 的 watch_ratio，因此 MVP 用户表示采用更朴素的方式：
    1. 最近历史 item 先由 MultimodalItemEncoder 编码到同一召回空间。
    2. 对有效历史做 masked mean pooling。
    3. 经过轻量 MLP 得到最终 user embedding。

后续如果引入时间间隔、评论或更细行为强度，可以在 pooling 层扩展权重。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedMeanPooling(nn.Module):
    """对 padding 后的历史 item embedding 做 mask 平均。"""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, history_item_embs: Tensor, history_mask: Tensor) -> Tensor:
        if history_item_embs.dim() != 3:
            raise ValueError(f"history_item_embs must be [B, L, D], got {tuple(history_item_embs.shape)}.")
        if history_mask.shape != history_item_embs.shape[:2]:
            raise ValueError(f"history_mask must be {tuple(history_item_embs.shape[:2])}, got {tuple(history_mask.shape)}.")

        mask = history_mask.to(dtype=history_item_embs.dtype, device=history_item_embs.device)
        weighted_sum = torch.sum(history_item_embs * mask.unsqueeze(-1), dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return weighted_sum / denom


class RecentHistoryUserTower(nn.Module):
    """基于最近多模态历史 item 的用户兴趣编码器。"""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("input_dim, hidden_dim and output_dim must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.output_dim = int(output_dim)
        self.normalize_output = bool(normalize_output)
        self.pooling = MaskedMeanPooling()
        self.user_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, history_item_embs: Tensor, history_mask: Tensor) -> Tensor:
        pooled = self.pooling(history_item_embs=history_item_embs, history_mask=history_mask)
        user_emb = self.user_projection(pooled)
        if self.normalize_output:
            user_emb = F.normalize(user_emb, dim=-1)
        return user_emb

    def encode_user(self, history_item_embs: Tensor, history_mask: Tensor) -> Tensor:
        """语义化别名，评估和训练统一调用 encode_user。"""
        return self.forward(history_item_embs=history_item_embs, history_mask=history_mask)
