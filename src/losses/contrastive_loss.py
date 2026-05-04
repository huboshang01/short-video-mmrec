"""
V2 双塔召回的对比学习损失。

本模块只关心一件事：给定同一个 batch 内的 user embedding 和目标 item embedding，
用 batch 内其他 item 作为负样本计算 InfoNCE。数据构造、模型编码和优化步骤都放在
上游模块中，避免 loss 层掺入训练流程细节。
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def info_nce_loss(
    user_embs: Tensor,
    target_item_embs: Tensor,
    temperature: float = 0.07,
    symmetric: bool = False,
):
    """
    计算双塔 InfoNCE loss，并返回训练日志需要的 batch 内检索指标。

    Args:
        user_embs: [B, D]，UserTower 输出的用户向量。
        target_item_embs: [B, D]，ItemEncoder 输出的正样本 item 向量。
        temperature: softmax 温度，越小分布越尖锐。
        symmetric: True 时同时计算 user->item 和 item->user 两个方向。

    Returns:
        loss: 可反传的标量 Tensor。
        metrics: 只用于日志展示的 Python float 指标。
    """
    if user_embs.dim() != 2 or target_item_embs.dim() != 2:
        raise ValueError(
            "user_embs and target_item_embs must be [B, D], "
            f"got {tuple(user_embs.shape)}, {tuple(target_item_embs.shape)}"
        )

    if user_embs.shape != target_item_embs.shape:
        raise ValueError(
            "user_embs and target_item_embs must have the same shape, "
            f"got {tuple(user_embs.shape)}, {tuple(target_item_embs.shape)}"
        )

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    # 双塔输出通常已经归一化；这里再做一次 L2 normalize，保证 loss 对输入更稳健。
    user_embs = F.normalize(user_embs, dim=-1)
    target_item_embs = F.normalize(target_item_embs, dim=-1)

    # logits[i, j] 表示第 i 个用户和第 j 个目标 item 的相似度。
    # 因为 dataset 按同一行返回 user 与其正样本 item，所以对角线 logits[i, i] 是正样本。
    logits = user_embs @ target_item_embs.t()
    logits = logits / temperature

    batch_size = user_embs.size(0)
    labels = torch.arange(batch_size, device=user_embs.device)

    loss_user_to_item = F.cross_entropy(logits, labels)

    if symmetric:
        # 反向把 item 当 query、user 当候选，约束两个塔在同一空间中互相可检索。
        loss_item_to_user = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_user_to_item + loss_item_to_user)
    else:
        loss_item_to_user = None
        loss = loss_user_to_item

    with torch.no_grad():
        # in-batch Acc@1/5：衡量每个 user 是否能在当前 batch 的候选 item 中找回正样本。
        acc1 = (logits.argmax(dim=1) == labels).float().mean()

        k = min(5, batch_size)
        topk = logits.topk(k=k, dim=1).indices
        acc5 = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()

        # 正样本 logit 与全局平均 logit 的间隔，可以辅助观察训练是否真的拉开正负样本。
        pos_logits = logits.diag().mean()
        all_logits = logits.mean()

    metrics = {
        "loss": loss.detach().item(),
        "loss_user_to_item": loss_user_to_item.detach().item(),
        "loss_item_to_user": (
            loss_item_to_user.detach().item() if loss_item_to_user is not None else None
        ),
        "inbatch_acc1": acc1.item(),
        "inbatch_acc5": acc5.item(),
        "pos_logits": pos_logits.item(),
        "all_logits": all_logits.item(),
        "temperature": float(temperature),
    }

    return loss, metrics
