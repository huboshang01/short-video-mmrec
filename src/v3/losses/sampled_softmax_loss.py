"""
V3 sampled softmax / InfoNCE 召回损失。

MicroLens-100K 只有隐式正反馈，没有显式负反馈，因此训练时为每个正样本采样 K 个
未交互 item，并让模型在 1+K 个候选中把正样本排到第 1 位。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sampled_softmax_loss(
    user_embs: Tensor,
    positive_item_embs: Tensor,
    negative_item_embs: Tensor,
    temperature: float = 0.07,
) -> tuple[Tensor, dict[str, float]]:
    """
    Args:
        user_embs:           [B, D]
        positive_item_embs:  [B, D]
        negative_item_embs:  [B, K, D]
        temperature:         logits 温度，越小区分度越强。
    """
    if user_embs.shape != positive_item_embs.shape:
        raise ValueError(f"user_embs and positive_item_embs shape mismatch: {user_embs.shape}, {positive_item_embs.shape}")
    if negative_item_embs.dim() != 3:
        raise ValueError(f"negative_item_embs must be [B, K, D], got {tuple(negative_item_embs.shape)}.")
    if negative_item_embs.shape[0] != user_embs.shape[0] or negative_item_embs.shape[2] != user_embs.shape[1]:
        raise ValueError("negative_item_embs must share batch size and dim with user_embs.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    pos_scores = torch.sum(user_embs * positive_item_embs, dim=-1, keepdim=True)
    neg_scores = torch.einsum("bd,bkd->bk", user_embs, negative_item_embs)
    logits = torch.cat([pos_scores, neg_scores], dim=1) / temperature

    labels = torch.zeros(user_embs.shape[0], dtype=torch.long, device=user_embs.device)
    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        mean_pos_score = pos_scores.mean()
        mean_neg_score = neg_scores.mean()

    metrics = {
        "loss": float(loss.detach().cpu()),
        "accuracy": float(accuracy.detach().cpu()),
        "pos_score": float(mean_pos_score.detach().cpu()),
        "neg_score": float(mean_neg_score.detach().cpu()),
        "score_margin": float((mean_pos_score - mean_neg_score).detach().cpu()),
    }
    return loss, metrics
