"""
V2.1 召回训练损失。

当前 MVP 使用 pointwise BCE：
    score(user, item) -> label

相比旧版 in-batch InfoNCE，BCE 的优势是：
    - 同一个用户可以拥有多个正样本，不要求 batch 内只有一个 item 是正确答案。
    - 显式负反馈可以直接进入训练，而不是把其他正样本误当负样本。
    - sample_weight 可以表达 watch_ratio 强度。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dot_product_scores(user_embs: Tensor, item_embs: Tensor, temperature: float = 1.0) -> Tensor:
    """计算 user/item 点积分数；输入通常已经 L2 normalize，点积等价 cosine。"""
    if user_embs.shape != item_embs.shape:
        raise ValueError(f"user_embs and item_embs must have same shape, got {user_embs.shape}, {item_embs.shape}.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    return torch.sum(user_embs * item_embs, dim=-1) / temperature


def bce_retrieval_loss(
    user_embs: Tensor,
    item_embs: Tensor,
    labels: Tensor,
    sample_weights: Tensor | None = None,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """
    pointwise BCE 召回损失。

    Args:
        user_embs:       [B, D]，UserTower 输出。
        item_embs:       [B, D]，ItemTower 输出。
        labels:          [B]，1 表示正反馈，0 表示显式负反馈。
        sample_weights:  [B]，样本权重；正样本可由 watch_ratio 强度决定。
        temperature:     点积分数温度，越小 sigmoid 前 logit 越尖锐。
    """
    if labels.dim() != 1:
        labels = labels.reshape(-1)
    labels = labels.to(dtype=user_embs.dtype, device=user_embs.device)

    scores = dot_product_scores(user_embs=user_embs, item_embs=item_embs, temperature=temperature)
    per_sample_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")

    if sample_weights is not None:
        weights = sample_weights.reshape(-1).to(dtype=user_embs.dtype, device=user_embs.device).clamp_min(0.0)
        loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        loss = per_sample_loss.mean()

    with torch.no_grad():
        probs = torch.sigmoid(scores)
        preds = (probs >= 0.5).to(dtype=labels.dtype) # 把布尔值转成和 labels 一样的数据类型
        accuracy = (preds == labels).float().mean()

        pos_mask = labels >= 0.5
        neg_mask = ~pos_mask # 布尔取反
        pos_score = scores[pos_mask].mean() if pos_mask.any() else scores.new_tensor(0.0)
        neg_score = scores[neg_mask].mean() if neg_mask.any() else scores.new_tensor(0.0)

    metrics = {
        "loss": float(loss.detach().cpu()),
        "accuracy": float(accuracy.detach().cpu()),
        "positive_ratio": float(labels.mean().detach().cpu()),
        "pos_score": float(pos_score.detach().cpu()),
        "neg_score": float(neg_score.detach().cpu()),
        "score_margin": float((pos_score - neg_score).detach().cpu()),
    }

    return loss, metrics
