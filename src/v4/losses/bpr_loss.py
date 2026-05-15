"""
V4 LightGCN 的 BPR 排序损失。

BPR（Bayesian Personalized Ranking）用于隐式反馈推荐场景：
    对同一个用户，模型应该让已交互正样本 item 的分数
    高于未交互负样本 item 的分数。

本模块只负责损失和训练日志指标，不关心模型结构或负采样逻辑。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def bpr_loss(pos_scores: Tensor, neg_scores: Tensor) -> tuple[Tensor, dict[str, float]]:
    """
    计算一组对齐正负样本的 BPR loss。

    Args:
        pos_scores: [B]，用户与正样本 item 的打分。
        neg_scores: [B]，同一批用户与负样本 item 的打分。
    """
    if pos_scores.shape != neg_scores.shape:
        raise ValueError(f"pos_scores and neg_scores shape mismatch: {pos_scores.shape}, {neg_scores.shape}")

    # score_margin 越大，说明正样本越明显排在负样本前面。
    score_margin = pos_scores - neg_scores
    # -log sigmoid(pos-neg)：当正样本分数高于负样本时 loss 会变小。
    loss = -F.logsigmoid(score_margin).mean()

    with torch.no_grad():
        # accuracy 表示当前 batch 中正样本分数高于负样本的比例。
        accuracy = (score_margin > 0).float().mean()

    # 这些指标用于训练日志观察，不参与反向传播。
    metrics = {
        "loss": float(loss.detach().cpu()),
        "accuracy": float(accuracy.detach().cpu()),
        "pos_score": float(pos_scores.mean().detach().cpu()),
        "neg_score": float(neg_scores.mean().detach().cpu()),
        "score_margin": float(score_margin.mean().detach().cpu()),
    }
    return loss, metrics
