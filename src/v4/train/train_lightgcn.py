"""
V4 LightGCN 的训练与 sampled validation 工具。

脚本层负责读取配置、构造 Dataset、模型和图邻接矩阵；本模块只关心：
    1. 单轮 BPR 训练；
    2. sampled negative 验证；
    3. checkpoint 保存。

注意：这里的验证不是 full-catalog 评估，只是在验证正样本 + 采样负样本上
观察 BPR 排序质量。完整 Recall@K 评估在 scripts/v4/02_eval_full_recall.py。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.v4.losses.bpr_loss import bpr_loss


# 训练和验证日志统一跟踪的指标名。
TRACKED_METRICS = ("loss", "accuracy", "pos_score", "neg_score", "score_margin")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """把 batch 中的小型 index tensor 移动到目标设备。"""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _new_metric_sums() -> dict[str, float]:
    """初始化 epoch 级别的指标累加器。"""
    return {metric_name: 0.0 for metric_name in TRACKED_METRICS}


def _accumulate_metrics(metric_sums: dict[str, float], metrics: dict[str, float], batch_size: int) -> None:
    """按样本数加权累加 batch 指标，避免最后一个小 batch 扭曲均值。"""
    for metric_name in TRACKED_METRICS:
        metric_sums[metric_name] += float(metrics[metric_name]) * batch_size


def _average_metrics(metric_sums: dict[str, float], sample_count: int) -> dict[str, float]:
    """把累加指标除以样本数，得到 epoch 平均指标。"""
    denom = max(sample_count, 1)
    return {metric_name: metric_sums[metric_name] / denom for metric_name in TRACKED_METRICS}


def _set_progress_metrics(pbar: tqdm, metric_sums: dict[str, float], sample_count: int) -> None:
    """把核心指标显示到 tqdm 进度条上，便于训练时观察趋势。"""
    metrics = _average_metrics(metric_sums, sample_count)
    pbar.set_postfix(
        loss=f"{metrics['loss']:.4f}",
        acc=f"{metrics['accuracy']:.4f}",
        margin=f"{metrics['score_margin']:.4f}",
    )


def train_one_epoch_bpr(
    model,
    norm_adj: torch.Tensor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 100,
) -> dict[str, float]:
    """训练一轮 LightGCN BPR。"""
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    model.train()
    metric_sums = _new_metric_sums()
    sample_count = 0
    pbar = tqdm(dataloader, desc="train", dynamic_ncols=True)

    for step, batch in enumerate(pbar, start=1):
        # batch 包含 user_index、positive_item_index、negative_item_index。
        batch = move_batch_to_device(batch, device)
        batch_size = int(batch["user_index"].numel())

        optimizer.zero_grad(set_to_none=True)
        # LightGCN 前向会先基于 norm_adj 编码全量 user/item，再取当前 batch 的正负 pair 打分。
        pos_scores, neg_scores = model(
            user_indices=batch["user_index"],
            positive_item_indices=batch["positive_item_index"],
            negative_item_indices=batch["negative_item_index"],
            norm_adj=norm_adj,
        )
        # BPR loss 推动 pos_scores 大于 neg_scores。
        loss, metrics = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        # 记录本 batch 对 epoch 平均指标的贡献。
        sample_count += batch_size
        _accumulate_metrics(metric_sums, metrics, batch_size=batch_size)
        if step % log_every == 0 or step == 1:
            _set_progress_metrics(pbar, metric_sums, sample_count)

    return _average_metrics(metric_sums, sample_count)


@torch.no_grad()
def evaluate_sampled_bpr(
    model,
    norm_adj: torch.Tensor,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """在采样负样本上验证 BPR 排序效果。"""
    model.eval()
    # 验证阶段图和参数都固定，一次性编码全量 embedding，减少重复图传播。
    user_embs, item_embs = model.encode_all(norm_adj)

    metric_sums = _new_metric_sums()
    sample_count = 0
    pbar = tqdm(dataloader, desc="eval", dynamic_ncols=True)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        # 与训练相同口径，只是不做反向传播。
        batch_user_embs = user_embs[batch["user_index"]]
        pos_scores = torch.sum(batch_user_embs * item_embs[batch["positive_item_index"]], dim=-1)
        neg_scores = torch.sum(batch_user_embs * item_embs[batch["negative_item_index"]], dim=-1)
        _, metrics = bpr_loss(pos_scores, neg_scores)

        batch_size = int(batch["user_index"].numel())
        sample_count += batch_size
        _accumulate_metrics(metric_sums, metrics, batch_size=batch_size)
        _set_progress_metrics(pbar, metric_sums, sample_count)

    return _average_metrics(metric_sums, sample_count)


def save_lightgcn_checkpoint(
    path: str | Path,
    model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """保存 LightGCN checkpoint，供继续训练或 full-catalog 评估复用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # config 内含模型结构、数据路径、用户/item 数和训练配置，评估脚本靠它复原模型。
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )
