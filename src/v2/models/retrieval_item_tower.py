"""
V2.1 召回 item tower。

ItemTower 的职责是把多种 item 特征融合到统一召回向量空间：
    - BGE item_text embedding: 内容语义，支持语义泛化和冷启动。
    - item_id embedding: 协同过滤记忆，学习高密度交互里的 item 偏好结构。
    - category embedding: 类目先验，让召回空间保留粗粒度兴趣信息。

MVP 采用 concat 融合而不是 add。因为文本、ID 和类目不是同构语义空间，concat 后交给
MLP 学习融合关系更稳，也方便后续继续拼接 image/OCR/ASR/audio 等多模态特征。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TextIdCategoryItemTower(nn.Module):
    """
    文本 + item_id + category 的轻量 item tower。

    输入：
        item_text_emb:     [..., text_dim]，BGE 预编码文本向量。
        item_indices:      [...]，item 在 embedding cache 中的行号，用于查 item_id embedding。
        category_indices:  [...]，category vocab 行号，用于查 category embedding。

    输出：
        item_emb:          [..., output_dim]，L2 normalize 后可直接做 dot/cosine 召回。
    """

    def __init__(
        self,
        text_dim: int,
        num_items: int,
        num_categories: int,
        text_proj_dim: int = 256,
        item_id_dim: int = 64,
        category_dim: int = 32,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
        id_feature_dropout: float = 0.1,
        normalize_output: bool = True,
        use_item_id: bool = True,
        use_category: bool = True,
    ) -> None:
        super().__init__()

        if text_dim <= 0 or num_items <= 0 or num_categories <= 0:
            raise ValueError("text_dim, num_items and num_categories must be positive.")
        if min(text_proj_dim, item_id_dim, category_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("all embedding/projection dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not 0 <= id_feature_dropout < 1:
            raise ValueError("id_feature_dropout must be in [0, 1).")

        self.text_dim = int(text_dim)
        self.num_items = int(num_items)
        self.num_categories = int(num_categories)
        self.output_dim = int(output_dim)
        self.normalize_output = bool(normalize_output)
        self.use_item_id = bool(use_item_id)
        self.use_category = bool(use_category)

        # 文本语义先降/投到较紧凑空间，避免 concat 后 MLP 参数过大。
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, text_proj_dim),
            nn.LayerNorm(text_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.use_item_id:
            self.item_embedding = nn.Embedding(num_items, item_id_dim)
            self.item_feature_dropout = nn.Dropout(id_feature_dropout)
        else:
            self.item_embedding = None
            self.item_feature_dropout = nn.Identity()

        if self.use_category:
            self.category_embedding = nn.Embedding(num_categories, category_dim)
        else:
            self.category_embedding = None

        fusion_dim = text_proj_dim
        if self.use_item_id:
            fusion_dim += item_id_dim
        if self.use_category:
            fusion_dim += category_dim

        # concat 后用 MLP 学习不同模态/特征之间的非线性融合关系。
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """对稀疏 ID 类 embedding 使用较小初始化，降低它们一开始压过文本语义的风险。"""
        if self.item_embedding is not None:
            nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        if self.category_embedding is not None:
            nn.init.normal_(self.category_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        item_text_emb: Tensor,
        item_indices: Tensor | None = None,
        category_indices: Tensor | None = None,
    ) -> Tensor:
        if item_text_emb.shape[-1] != self.text_dim:
            raise ValueError(f"item_text_emb last dim must be {self.text_dim}, got {item_text_emb.shape[-1]}.")

        original_shape = item_text_emb.shape[:-1]
        flat_text = item_text_emb.reshape(-1, self.text_dim).float()

        features = [self.text_projection(flat_text)]

        if self.use_item_id:
            if item_indices is None:
                raise ValueError("item_indices is required when use_item_id=True.")
            flat_item_indices = item_indices.reshape(-1).long().to(item_text_emb.device)
            item_features = self.item_embedding(flat_item_indices)
            # 对 item_id 特征单独 dropout，防止模型只记 ID 而忽略文本语义。
            features.append(self.item_feature_dropout(item_features))

        if self.use_category:
            if category_indices is None:
                raise ValueError("category_indices is required when use_category=True.")
            flat_category_indices = category_indices.reshape(-1).long().to(item_text_emb.device)
            features.append(self.category_embedding(flat_category_indices))

        fused = torch.cat(features, dim=-1)
        item_emb = self.fusion(fused)

        if self.normalize_output:
            item_emb = F.normalize(item_emb, dim=-1)

        return item_emb.reshape(*original_shape, self.output_dim)

    def encode_item(
        self,
        item_text_emb: Tensor,
        item_indices: Tensor | None = None,
        category_indices: Tensor | None = None,
    ) -> Tensor:
        """语义化别名，训练和评估代码统一调用 encode_item。"""
        return self.forward(item_text_emb, item_indices=item_indices, category_indices=category_indices)
