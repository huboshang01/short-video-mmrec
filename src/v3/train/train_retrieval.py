"""
V3 sampled-softmax 双塔召回训练工具。

本模块只包含训练/验证循环和 checkpoint 保存，不处理命令行参数。脚本层负责构造
Dataset、FeatureCache 和模型；这里负责把 item index batch 转成多模态特征并执行前向。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.v3.data.microlens_retrieval_dataset import MultimodalFeatureCache
from src.v3.losses.sampled_softmax_loss import sampled_softmax_loss


TRACKED_METRICS = ("loss", "accuracy", "pos_score", "neg_score", "score_margin")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """只移动小型 index/mask Tensor；大特征由 FeatureCache 按需 gather。"""
    # Dataset 只返回 index/mask，因此这里不会搬运 text/image/video 大矩阵。
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def encode_retrieval_batch(
    item_tower,
    user_tower,
    feature_cache: MultimodalFeatureCache,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """统一 V3 双塔前向路径。"""
    # batch 中只有 item index；真正的多模态特征在下面通过 FeatureCache 拉取。
    history_indices = batch["history_item_indices"]
    target_indices = batch["target_item_index"]
    negative_indices = batch["negative_item_indices"]

    # 历史 item: index -> text/image/video 特征 -> item tower -> history item embeddings。
    history_features = feature_cache.gather(history_indices, device=device)
    history_item_embs = item_tower.encode_item(**history_features, item_indices=history_indices)
    # user tower 使用 mask 忽略 padding，只聚合真实历史 item。
    user_embs = user_tower.encode_user(
        history_item_embs=history_item_embs,
        history_mask=batch["history_mask"],
    )

    # 正样本 item 编码，形状通常为 [B, D]。
    target_features = feature_cache.gather(target_indices, device=device)
    positive_item_embs = item_tower.encode_item(**target_features, item_indices=target_indices)

    # 负样本 item 编码，形状通常为 [B, K, D]。
    negative_features = feature_cache.gather(negative_indices, device=device)
    negative_item_embs = item_tower.encode_item(**negative_features, item_indices=negative_indices)
    return user_embs, positive_item_embs, negative_item_embs


def _new_metric_sums() -> dict[str, float]:
    return {metric_name: 0.0 for metric_name in TRACKED_METRICS}


def _accumulate_metrics(metric_sums: dict[str, float], metrics: dict[str, float], batch_size: int) -> None:
    """按样本数加权累加，避免最后一个小 batch 扭曲 epoch 均值。"""
    for metric_name in TRACKED_METRICS:
        # batch 指标先乘 batch_size，最后再除总样本数得到 epoch 平均。
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


def train_one_epoch_sampled_softmax(
    item_tower,
    user_tower,
    feature_cache: MultimodalFeatureCache,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 0.07,
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
        batch_size = int(batch["target_item_index"].numel())

        optimizer.zero_grad(set_to_none=True)
        # 将 history/target/negative 的 index batch 编码成三组召回向量。
        user_embs, positive_item_embs, negative_item_embs = encode_retrieval_batch(
            item_tower=item_tower,
            user_tower=user_tower,
            feature_cache=feature_cache,
            batch=batch,
            device=device,
        )
        # sampled softmax: 每个用户在 1 个正样本 + K 个负样本中识别正样本。
        loss, metrics = sampled_softmax_loss(
            user_embs=user_embs,
            positive_item_embs=positive_item_embs,
            negative_item_embs=negative_item_embs,
            temperature=temperature,
        )
        # 反向传播同时更新 item tower 和 user tower。
        loss.backward()
        optimizer.step()

        sample_count += batch_size
        _accumulate_metrics(metric_sums, metrics, batch_size=batch_size)
        if step % log_every == 0 or step == 1:
            _set_progress_metrics(pbar, metric_sums, sample_count)

    return _average_metrics(metric_sums, sample_count)


@torch.no_grad()
def evaluate_sampled_softmax(
    item_tower,
    user_tower,
    feature_cache: MultimodalFeatureCache,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 0.07,
) -> dict[str, float]:
    item_tower.eval()
    user_tower.eval()

    metric_sums = _new_metric_sums()
    sample_count = 0
    pbar = tqdm(dataloader, desc="eval", dynamic_ncols=True)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        batch_size = int(batch["target_item_index"].numel())
        # 验证阶段复用同一条前向路径，但 no_grad + eval 模式不更新参数。
        user_embs, positive_item_embs, negative_item_embs = encode_retrieval_batch(
            item_tower=item_tower,
            user_tower=user_tower,
            feature_cache=feature_cache,
            batch=batch,
            device=device,
        )
        _, metrics = sampled_softmax_loss(
            user_embs=user_embs,
            positive_item_embs=positive_item_embs,
            negative_item_embs=negative_item_embs,
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
    """保存 V3 召回模型 checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # checkpoint 保存模型权重、优化器状态、指标和完整配置，便于继续训练或独立评估。
    torch.save(
        {
            "epoch": int(epoch),
            "item_tower": item_tower.state_dict(),
            "user_tower": user_tower.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )
