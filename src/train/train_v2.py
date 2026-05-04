"""
V2 双塔 InfoNCE 训练工具。

本模块承接 DataLoader 输出的 batch，调用共享 ItemEncoder 与 UserTower 完成前向，
再把 user/item 向量交给 contrastive_loss 计算 batch 内对比学习目标。脚本层负责
参数解析和模型实例化，这里只保留训练、验证和 checkpoint 保存的通用逻辑。
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses.contrastive_loss import info_nce_loss


TRACKED_METRICS = ("loss", "inbatch_acc1", "inbatch_acc5")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """把 batch 中的 Tensor 搬到训练设备，非 Tensor 元信息保持原样。"""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _encode_batch(
    item_encoder,
    user_tower,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """复用双塔前向：历史 item 文本向量走 UserTower，target item 文本向量走 ItemEncoder。"""
    user_embs = user_tower(
        history_item_text_embs=batch["history_item_text_embs"],
        watch_ratios=batch["watch_ratios"],
        mask=batch["mask"],
    )
    target_item_embs = item_encoder.encode_item(batch["target_item_text_emb"])
    return user_embs, target_item_embs


def _new_metric_sums() -> dict[str, float]:
    """初始化 epoch 级指标累加器。"""
    return {metric_name: 0.0 for metric_name in TRACKED_METRICS}


def _accumulate_metrics(
    metric_sums: dict[str, float],
    batch_metrics: dict[str, float | None],
) -> None:
    """只累加 epoch 汇总需要的指标，避免日志字段变化影响训练主循环。"""
    for metric_name in TRACKED_METRICS:
        metric_sums[metric_name] += float(batch_metrics[metric_name])


def _average_metrics(metric_sums: dict[str, float], steps: int) -> dict[str, float]:
    """把累加指标转成 epoch 平均值；空 dataloader 时避免除零。"""
    denom = max(steps, 1)
    return {metric_name: metric_sums[metric_name] / denom for metric_name in TRACKED_METRICS}


def _set_progress_metrics(pbar: tqdm, metric_sums: dict[str, float], steps: int) -> None:
    """统一 train/eval 进度条展示，减少两处循环中的重复格式化代码。"""
    metrics = _average_metrics(metric_sums, steps)
    pbar.set_postfix(
        loss=f"{metrics['loss']:.4f}",
        acc1=f"{metrics['inbatch_acc1']:.4f}",
        acc5=f"{metrics['inbatch_acc5']:.4f}",
    )


def train_one_epoch(
    item_encoder,
    user_tower,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    temperature: float = 0.07,
    symmetric_loss: bool = False,
    log_every: int = 100,
):
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    item_encoder.train()
    user_tower.train()

    metric_sums = _new_metric_sums()
    total_steps = 0

    pbar = tqdm(dataloader, desc="train", dynamic_ncols=True)

    for step, batch in enumerate(pbar, start=1):
        batch = move_batch_to_device(batch, device)

        # 清空上一轮梯度后再前向，set_to_none=True 可以减少一次显式置零的内存写入。
        optimizer.zero_grad(set_to_none=True)
        user_embs, target_item_embs = _encode_batch(item_encoder, user_tower, batch)

        loss, metrics = info_nce_loss(
            user_embs=user_embs,
            target_item_embs=target_item_embs,
            temperature=temperature,
            symmetric=symmetric_loss,
        )

        loss.backward()
        optimizer.step()

        _accumulate_metrics(metric_sums, metrics)
        total_steps += 1

        if step % log_every == 0 or step == 1:
            _set_progress_metrics(pbar, metric_sums, total_steps)

    return _average_metrics(metric_sums, total_steps)


@torch.no_grad()
def evaluate_inbatch(
    item_encoder,
    user_tower,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 0.07,
    symmetric_loss: bool = False,
):
    item_encoder.eval()
    user_tower.eval()

    metric_sums = _new_metric_sums()
    total_steps = 0

    pbar = tqdm(dataloader, desc="eval", dynamic_ncols=True)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        # 验证阶段沿用同一套双塔编码路径，确保评估口径和训练完全一致。
        user_embs, target_item_embs = _encode_batch(item_encoder, user_tower, batch)

        _, metrics = info_nce_loss(
            user_embs=user_embs,
            target_item_embs=target_item_embs,
            temperature=temperature,
            symmetric=symmetric_loss,
        )

        _accumulate_metrics(metric_sums, metrics)
        total_steps += 1

        _set_progress_metrics(pbar, metric_sums, total_steps)

    return _average_metrics(metric_sums, total_steps)


def save_checkpoint(
    path: str | Path,
    item_encoder,
    user_encoder,
    optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # checkpoint 拆开保存 item_encoder 和 user_encoder，加载时可重建共享 ItemEncoder 的 UserTower。
    checkpoint = {
        "epoch": epoch,
        "item_encoder": item_encoder.state_dict(),
        "user_encoder": user_encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }

    torch.save(checkpoint, path)
