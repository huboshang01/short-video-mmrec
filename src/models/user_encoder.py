import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.behavior_adapter import ProjectionHead
from src.models.item_encoder import ItemEncoder


class WatchRatioWeightedPooling(nn.Module):
    """
    watch_ratio 加权池化。把用户多个历史 item embedding，按 watch_ratio 加权压成一个用户兴趣向量。

    输入：
        history_item_embs: [B, L, D]，用户历史 item 的行为感知向量
        watch_ratios:      [B, L]，对应历史 item 的观看比例
        mask:              [B, L]，1 表示有效历史，0 表示 padding

    输出：
        pooled_user_emb:   [B, D]，加权聚合后的用户兴趣向量
    """

    def __init__(self, weight_cap: float | None = 5.0, eps: float = 1e-8):
        super().__init__()
        if weight_cap is not None and weight_cap <= 0:
            raise ValueError("weight_cap must be positive or None.")
        self.weight_cap = weight_cap
        self.eps = eps

    def forward(self, history_item_embs: Tensor, watch_ratios: Tensor, mask: Tensor | None = None) -> Tensor:
        if history_item_embs.dim() != 3:
            raise ValueError(f"history_item_embs must be [B, L, D], got {tuple(history_item_embs.shape)}")
        if watch_ratios.dim() != 2:
            raise ValueError(f"watch_ratios must be [B, L], got {tuple(watch_ratios.shape)}")

        batch_size, seq_len, _ = history_item_embs.shape
        if watch_ratios.shape != (batch_size, seq_len):
            raise ValueError(f"watch_ratios shape must be {(batch_size, seq_len)}, got {tuple(watch_ratios.shape)}")

        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=history_item_embs.dtype, device=history_item_embs.device)
        else:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(f"mask shape must be {(batch_size, seq_len)}, got {tuple(mask.shape)}")
            mask = mask.to(dtype=history_item_embs.dtype, device=history_item_embs.device)

        watch_ratios = watch_ratios.to(dtype=history_item_embs.dtype, device=history_item_embs.device)

        # watch_ratio 作为兴趣强度权重；负数无意义，极端大值做截断，避免单条历史支配用户向量
        weights = torch.clamp(watch_ratios, min=0.0)
        if self.weight_cap is not None:
            weights = torch.clamp(weights, max=self.weight_cap)
        weights = weights * mask

        weight_sum = weights.sum(dim=1, keepdim=True)

        # 如果某个用户有效权重全为 0，则退化为对有效历史做均匀平均
        valid_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        uniform_weights = mask / valid_count

        normalized_weights = weights / weight_sum.clamp_min(self.eps)
        use_uniform = (weight_sum <= self.eps).to(dtype=history_item_embs.dtype)
        final_weights = use_uniform * uniform_weights + (1.0 - use_uniform) * normalized_weights

        return torch.sum(history_item_embs * final_weights.unsqueeze(-1), dim=1)


class UserEncoder(nn.Module):
    """
    V2 user 编码器。接收历史 item 向量序列，先做 watch_ratio 加权池化，
    得到池化后的用户兴趣向量，再经过可训练投影头得到最终 user_embedding。

    输入：
        history_item_embs: 已经经过共享 ItemEncoder 的历史 item 向量 [B, L, D]
        watch_ratios:      历史 item 对应的观看比例 [B, L]
        mask:              历史序列 padding mask [B, L]

    输出：
        user_embedding:    投影后的行为感知用户兴趣向量 [B, D]
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
        use_input_projection: bool = False,
        normalize_output: bool = True,
        weight_cap: float | None = 5.0,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_output = normalize_output

        self.pooling = WatchRatioWeightedPooling(weight_cap=weight_cap)
        self.projection = ProjectionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
            use_input_projection=use_input_projection,
        )

    def forward(self, history_item_embs: Tensor, watch_ratios: Tensor, mask: Tensor | None = None) -> Tensor:
        """接收历史 item 向量序列，池化后再映射为 user embedding。"""
        user_emb = self.pooling(history_item_embs=history_item_embs, watch_ratios=watch_ratios, mask=mask)
        user_emb = self.projection(user_emb)

        if self.normalize_output:
            user_emb = F.normalize(user_emb, dim=-1)

        return user_emb

    def encode_user(self, history_item_embs: Tensor, watch_ratios: Tensor, mask: Tensor | None = None) -> Tensor:
        """和 forward 等价，便于训练/评估代码语义化调用。"""
        return self.forward(history_item_embs, watch_ratios, mask)


class UserTower(nn.Module):
    """
    把历史 item 的 BGE 文本向量，先送进共享 ItemEncoder，再交给 UserEncoder。
    
    用户塔：历史 item 文本向量 -> 共享 ItemEncoder -> UserEncoder -> user embedding。

    关键设计：
        history item 和 target item 应传入同一个 ItemEncoder 实例，
        这样 InfoNCE 训练时二者会被拉到同一个 behavior-aware semantic space。
    """

    def __init__(self, item_encoder: ItemEncoder, user_encoder: UserEncoder):
        super().__init__()
        self.item_encoder = item_encoder
        self.user_encoder = user_encoder

    def forward(self, history_item_text_embs: Tensor, watch_ratios: Tensor, mask: Tensor | None = None) -> Tensor:
        if history_item_text_embs.dim() != 3:
            raise ValueError(f"history_item_text_embs must be [B, L, D], got {tuple(history_item_text_embs.shape)}")

        batch_size, seq_len, text_dim = history_item_text_embs.shape

        # 将 [B, L, D] 展平成 [B*L, D]，复用 item tower 的 ItemEncoder 编码历史 item
        flat_history = history_item_text_embs.reshape(batch_size * seq_len, text_dim)
        flat_item_embs = self.item_encoder.encode_item(flat_history)
        history_item_embs = flat_item_embs.reshape(batch_size, seq_len, -1)

        return self.user_encoder.encode_user(history_item_embs, watch_ratios, mask)

    def encode_user(self, history_item_text_embs: Tensor, watch_ratios: Tensor, mask: Tensor | None = None) -> Tensor:
        """和 forward 等价，输入历史 item 文本向量，输出 user embedding。"""
        return self.forward(history_item_text_embs, watch_ratios, mask)
