"""
V4 协同过滤召回模型：LightGCN。

LightGCN 把推荐数据看成 user-item 二部图，只保留两类核心信息：
    1. user/item 的 ID embedding；
    2. 基于交互图邻接矩阵的多层传播。

相比普通 GCN，它去掉特征变换和非线性激活，更适合召回阶段的协同过滤建模。
V3 负责多模态内容语义，V4 负责用户行为图上的协同信号。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LightGCN(nn.Module):
    """LightGCN 编码器：聚合第 0 层到第 L 层的 user/item embedding。"""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        num_layers: int = 3,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        # 基础维度必须为正；num_layers=0 时退化为纯 ID embedding 矩阵分解。
        if min(num_users, num_items, embedding_dim) <= 0:
            raise ValueError("num_users, num_items and embedding_dim must be positive.")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative.")

        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.normalize_output = bool(normalize_output)

        # LightGCN 不使用内容特征，只学习 user ID 和 item ID 两张 embedding 表。
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化 user/item embedding，std=0.1 是 LightGCN 常见的小随机初始化。"""
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)

    def encode_all(self, norm_adj: Tensor) -> tuple[Tensor, Tensor]:
        """
        对全量用户和 item 做 LightGCN 图传播。

        Args:
            norm_adj: 归一化后的 user-item 二部图邻接矩阵。

        Returns:
            user_embs: [num_users, D]
            item_embs: [num_items, D]
        """
        # 把 user 和 item 拼成一张大节点表，便于直接和二部图邻接矩阵相乘。
        all_embs = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        # 第 0 层是不经过图传播的原始 ID embedding。
        layer_embs = [all_embs]
        for _ in range(self.num_layers):
            # 稀疏矩阵乘法完成一次邻居信息传播。
            all_embs = torch.sparse.mm(norm_adj, all_embs)
            layer_embs.append(all_embs)

        # LightGCN 使用所有层 embedding 的平均值作为最终表示，避免只依赖最后一层。
        final_embs = torch.stack(layer_embs, dim=0).mean(dim=0)
        # 拼接表按 [users, items] 顺序排列，这里再拆回两张 embedding 表。
        user_embs, item_embs = torch.split(final_embs, [self.num_users, self.num_items], dim=0)
        if self.normalize_output:
            # 可选 L2 normalize，便于做 cosine 风格的内积打分。
            user_embs = F.normalize(user_embs, dim=-1)
            item_embs = F.normalize(item_embs, dim=-1)
        return user_embs, item_embs

    def score_pairs(self, user_indices: Tensor, item_indices: Tensor, norm_adj: Tensor) -> Tensor:
        """对一组对齐的 user-item pair 做点积打分。"""
        user_embs, item_embs = self.encode_all(norm_adj)
        return torch.sum(user_embs[user_indices] * item_embs[item_indices], dim=-1)

    def forward(
        self,
        user_indices: Tensor,
        positive_item_indices: Tensor,
        negative_item_indices: Tensor,
        norm_adj: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """训练前向：输出正样本分数和负样本分数，供 BPR loss 使用。"""
        user_embs, item_embs = self.encode_all(norm_adj)
        # 取当前 batch 的用户 embedding，形状 [B, D]。
        batch_user_embs = user_embs[user_indices]
        # 正负 item 都与同一个 batch_user_embs 做点积，得到 [B] 分数。
        pos_scores = torch.sum(batch_user_embs * item_embs[positive_item_indices], dim=-1)
        neg_scores = torch.sum(batch_user_embs * item_embs[negative_item_indices], dim=-1)
        return pos_scores, neg_scores


def build_normalized_adj(
    num_users: int,
    num_items: int,
    user_indices: Tensor,
    item_indices: Tensor,
    device: torch.device,
) -> Tensor:
    """
    为 user-item 二部图构造 LightGCN 使用的归一化邻接矩阵。

    节点编号约定：
        user 节点占用 [0, num_users)
        item 节点占用 [num_users, num_users + num_items)

    返回矩阵为 D^-1/2 A D^-1/2，适合做对称归一化图传播。
    """
    if user_indices.numel() != item_indices.numel():
        raise ValueError("user_indices and item_indices must have the same length.")
    if user_indices.numel() == 0:
        raise ValueError("Cannot build LightGCN adjacency from zero edges.")

    user_indices = user_indices.to(device=device, dtype=torch.long)
    # item 节点编号整体右移 num_users，避免和 user 节点编号冲突。
    item_nodes = item_indices.to(device=device, dtype=torch.long) + int(num_users)
    # 二部图是无向图：一条 user->item 边同时补 item->user 反向边。
    edge_rows = torch.cat([user_indices, item_nodes], dim=0)
    edge_cols = torch.cat([item_nodes, user_indices], dim=0)
    edge_values = torch.ones(edge_rows.numel(), dtype=torch.float32, device=device)

    num_nodes = int(num_users) + int(num_items)
    # degree[i] 表示节点 i 的邻居数量，用于后续对边权重做对称归一化。
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.scatter_add_(0, edge_rows, edge_values)
    degree = degree.clamp_min(1.0)
    # 每条边的归一化权重：1 / sqrt(deg(src) * deg(dst))。
    norm_values = degree[edge_rows].pow(-0.5) * degree[edge_cols].pow(-0.5)

    # 使用稀疏 COO 矩阵，避免为 10 万用户 x 近 2 万 item 构造巨大稠密矩阵。
    indices = torch.stack([edge_rows, edge_cols], dim=0)
    norm_adj = torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes), device=device)
    return norm_adj.coalesce()
