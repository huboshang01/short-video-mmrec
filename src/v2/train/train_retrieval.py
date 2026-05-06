"""
V2.1 pointwise retrieval 训练工具。

本模块只放训练/验证循环和 checkpoint 保存，不处理命令行参数和 Dataset 构造。
这样脚本层可以自由组合 small/big 数据源，而训练主逻辑保持稳定。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.v2.losses.retrieval_loss import bce_retrieval_loss


TRACKED_METRICS = ("loss", "accuracy", "positive_ratio", "pos_score", "neg_score", "score_margin")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """把 batch 中所有 Tensor 移到训练设备，非 Tensor 字段保持不变。"""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def encode_retrieval_batch(item_tower, user_tower, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """统一双塔前向路径：history 走 user tower，target 走 item tower。"""
    user_embs = user_tower.encode_user(
        history_item_text_embs=batch["history_item_text_embs"],
        history_item_indices=batch["history_item_indices"],
        history_category_indices=batch["history_category_indices"],
        history_watch_ratios=batch["history_watch_ratios"],
        history_time_deltas=batch["history_time_deltas"],
        history_mask=batch["history_mask"],
    )
    target_item_embs = item_tower.encode_item(
        item_text_emb=batch["target_item_text_emb"],
        item_indices=batch["target_item_index"],
        category_indices=batch["target_category_index"],
    )
    return user_embs, target_item_embs


def _new_metric_sums() -> dict[str, float]:
    return {metric_name: 0.0 for metric_name in TRACKED_METRICS}


def _accumulate_metrics(metric_sums: dict[str, float], metrics: dict[str, float], batch_size: int) -> None:
    """按样本数加权累加，避免最后一个小 batch 对 epoch 均值影响过大。"""
    for metric_name in TRACKED_METRICS:
        metric_sums[metric_name] += float(metrics[metric_name]) * batch_size


def _average_metrics(metric_sums: dict[str, float], sample_count: int) -> dict[str, float]:
    denom = max(sample_count, 1)
    return {metric_name: metric_sums[metric_name] / denom for metric_name in TRACKED_METRICS}


def _set_progress_metrics(pbar: tqdm, metric_sums: dict[str, float], sample_count: int) -> None:
    metrics = _average_metrics(metric_sums, sample_count)
    pbar.set_postfix(
        loss=f"{metrics['loss']:.4f}",
        acc=f"{metrics['accuracy']:.4f}",
        margin=f"{metrics['score_margin']:.4f}",
    )


def train_one_epoch_bce(
    item_tower,
    user_tower,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 1.0,
    log_every: int = 100,
) -> dict[str, float]:
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    item_tower.train()
    user_tower.train()

    metric_sums = _new_metric_sums()
    sample_count = 0
    pbar = tqdm(dataloader, desc="train", dynamic_ncols=True)

    for step, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, device)
        batch_size = int(batch["label"].numel())

        optimizer.zero_grad(set_to_none=True)
        user_embs, target_item_embs = encode_retrieval_batch(item_tower, user_tower, batch)
        loss, metrics = bce_retrieval_loss(
            user_embs=user_embs,
            item_embs=target_item_embs,
            labels=batch["label"],
            sample_weights=batch["sample_weight"],
            temperature=temperature,
        )
        loss.backward()
        optimizer.step()

        sample_count += batch_size
        _accumulate_metrics(metric_sums, metrics, batch_size=batch_size)

        if step % log_every == 0 or step == 1:
            _set_progress_metrics(pbar, metric_sums, sample_count)

    return _average_metrics(metric_sums, sample_count)


@torch.no_grad()
def evaluate_pointwise_bce(
    item_tower,
    user_tower,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> dict[str, float]:
    item_tower.eval()
    user_tower.eval()

    metric_sums = _new_metric_sums()
    sample_count = 0
    pbar = tqdm(dataloader, desc="eval", dynamic_ncols=True)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        batch_size = int(batch["label"].numel())

        user_embs, target_item_embs = encode_retrieval_batch(item_tower, user_tower, batch)
        _, metrics = bce_retrieval_loss(
            user_embs=user_embs,
            item_embs=target_item_embs,
            labels=batch["label"],
            sample_weights=batch["sample_weight"],
            temperature=temperature,
        )

        sample_count += batch_size
        _accumulate_metrics(metric_sums, metrics, batch_size=batch_size)
        _set_progress_metrics(pbar, metric_sums, sample_count)

    return _average_metrics(metric_sums, sample_count)


def save_retrieval_checkpoint(
    path: str | Path,
    item_tower,
    user_tower,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """保存召回模型 checkpoint；config 中包含 category vocab 和特征维度，便于评估脚本恢复。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "item_tower": item_tower.state_dict(),
        "user_tower": user_tower.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(checkpoint, path)
