"""
V2.1 召回 user tower。

MVP 采用工业推荐里常见的“最近行为 + 行为强度 + 时间衰减”用户表示：
    1. 最近 history item 先经过共享 ItemTower，进入同一个召回向量空间。
    2. watch_ratio 表示兴趣强度，time_delta 表示行为新鲜度。
    3. 加权池化后再经过轻量 MLP，得到最终 user embedding。

这比旧 V2 的 watch_ratio top50 全局兴趣更接近线上召回：用户画像只来自过去历史，
且最近行为会更重要。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.v2.models.retrieval_item_tower import TextIdCategoryItemTower


class WatchTimeWeightedPooling(nn.Module):
    """
    watch_ratio 与时间衰减联合加权池化。

    输入：
        history_item_embs:     [B, L, D]
        history_watch_ratios:  [B, L]
        history_time_deltas:   [B, L]，单位通常是秒，越小代表越新。
        history_mask:          [B, L]，1 表示有效历史，0 表示 padding。
    """

    def __init__(
        self,
        watch_weight_cap: float = 5.0,
        time_decay_days: float = 7.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if watch_weight_cap <= 0:
            raise ValueError("watch_weight_cap must be positive.")
        if time_decay_days <= 0:
            raise ValueError("time_decay_days must be positive.")

        self.watch_weight_cap = float(watch_weight_cap)
        self.time_decay_seconds = float(time_decay_days) * 24.0 * 3600.0
        self.eps = float(eps)

    def forward(
        self,
        history_item_embs: Tensor,
        history_watch_ratios: Tensor,
        history_time_deltas: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        if history_item_embs.dim() != 3:
            raise ValueError(f"history_item_embs must be [B, L, D], got {tuple(history_item_embs.shape)}.")

        batch_size, seq_len, _ = history_item_embs.shape
        expected_shape = (batch_size, seq_len)
        for name, value in {
            "history_watch_ratios": history_watch_ratios,
            "history_time_deltas": history_time_deltas,
            "history_mask": history_mask,
        }.items():
            if value.shape != expected_shape:
                raise ValueError(f"{name} must be {expected_shape}, got {tuple(value.shape)}.")

        dtype = history_item_embs.dtype
        device = history_item_embs.device
        watch_ratios = history_watch_ratios.to(dtype=dtype, device=device)
        time_deltas = history_time_deltas.to(dtype=dtype, device=device)
        mask = history_mask.to(dtype=dtype, device=device)

        # watch_ratio 越高权重越大；极端值截断，避免异常长播放支配用户向量。
        watch_weights = torch.clamp(watch_ratios, min=0.0, max=self.watch_weight_cap)

        # 越近的行为 time_delta 越小，exp 衰减后权重越大。
        time_weights = torch.exp(-torch.clamp(time_deltas, min=0.0) / self.time_decay_seconds)

        weights = watch_weights * time_weights * mask
        weight_sum = weights.sum(dim=1, keepdim=True)

        # 如果有效权重全为 0，退化成有效历史的均匀平均，避免 NaN。
        valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        uniform_weights = mask / valid_count
        normalized_weights = weights / weight_sum.clamp_min(self.eps)
        use_uniform = (weight_sum <= self.eps).to(dtype=dtype)
        final_weights = use_uniform * uniform_weights + (1.0 - use_uniform) * normalized_weights

        return torch.sum(history_item_embs * final_weights.unsqueeze(-1), dim=1)


class RecentHistoryUserTower(nn.Module):
    """
    基于最近历史行为的 user tower。

    UserTower 持有共享 ItemTower：history item 和 target item 使用同一套 item 表征，
    但训练目标从旧版 InfoNCE 改成 pointwise BCE，避免“一个 user 只能对应一个正 item”的限制。
    """

    def __init__(
        self,
        item_tower: TextIdCategoryItemTower,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
        watch_weight_cap: float = 5.0,
        time_decay_days: float = 7.0,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("input_dim, hidden_dim and output_dim must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.item_tower = item_tower
        self.output_dim = int(output_dim)
        self.normalize_output = bool(normalize_output)

        self.pooling = WatchTimeWeightedPooling(
            watch_weight_cap=watch_weight_cap,
            time_decay_days=time_decay_days,
        )
        self.user_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        history_item_text_embs: Tensor,
        history_item_indices: Tensor,
        history_category_indices: Tensor,
        history_watch_ratios: Tensor,
        history_time_deltas: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        # 先把历史 item 编到召回空间，再做用户兴趣聚合。
        history_item_embs = self.item_tower.encode_item(
            item_text_emb=history_item_text_embs,
            item_indices=history_item_indices,
            category_indices=history_category_indices,
        )

        pooled_user_emb = self.pooling(
            history_item_embs=history_item_embs,
            history_watch_ratios=history_watch_ratios,
            history_time_deltas=history_time_deltas,
            history_mask=history_mask,
        )
        user_emb = self.user_projection(pooled_user_emb)

        if self.normalize_output:
            user_emb = F.normalize(user_emb, dim=-1)

        return user_emb

    def encode_user(
        self,
        history_item_text_embs: Tensor,
        history_item_indices: Tensor,
        history_category_indices: Tensor,
        history_watch_ratios: Tensor,
        history_time_deltas: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        """语义化别名，评估和线上召回统一调用 encode_user。"""
        return self.forward(
            history_item_text_embs=history_item_text_embs,
            history_item_indices=history_item_indices,
            history_category_indices=history_category_indices,
            history_watch_ratios=history_watch_ratios,
            history_time_deltas=history_time_deltas,
            history_mask=history_mask,
        )
