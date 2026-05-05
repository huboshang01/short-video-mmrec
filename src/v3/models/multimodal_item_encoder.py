"""
V3 多模态 item encoder。

本模块把 MicroLens 官方预提取的 text / image / video 特征融合到统一召回空间：
    - text:  BGE-M3 标题文本特征。
    - image: CLIP-RN50 封面/图像特征。
    - video: VideoMAE 视频内容特征。

MVP 使用 concat + MLP 的稳健融合方式，保持和 V2 item tower 类似的工程风格。
后续可在同一接口下替换为 gated fusion、modality dropout 或 attention pooling。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultimodalItemEncoder(nn.Module):
    """text/image/video + 可选 item_id embedding 的轻量 item tower。"""

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        video_dim: int,
        num_items: int,
        text_proj_dim: int = 256,
        image_proj_dim: int = 256,
        video_proj_dim: int = 256,
        item_id_dim: int = 64,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
        id_feature_dropout: float = 0.1,
        use_item_id: bool = True,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        if min(text_dim, image_dim, video_dim, num_items) <= 0:
            raise ValueError("feature dimensions and num_items must be positive.")
        if min(text_proj_dim, image_proj_dim, video_proj_dim, item_id_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("projection/model dimensions must be positive.")
        if not 0 <= dropout < 1 or not 0 <= id_feature_dropout < 1:
            raise ValueError("dropout values must be in [0, 1).")

        self.text_dim = int(text_dim)
        self.image_dim = int(image_dim)
        self.video_dim = int(video_dim)
        self.num_items = int(num_items)
        self.output_dim = int(output_dim)
        self.use_item_id = bool(use_item_id)
        self.normalize_output = bool(normalize_output)

        self.text_projection = build_projection(text_dim, text_proj_dim, dropout)
        self.image_projection = build_projection(image_dim, image_proj_dim, dropout)
        self.video_projection = build_projection(video_dim, video_proj_dim, dropout)

        fusion_dim = text_proj_dim + image_proj_dim + video_proj_dim
        if self.use_item_id:
            self.item_embedding = nn.Embedding(num_items, item_id_dim)
            self.item_feature_dropout = nn.Dropout(id_feature_dropout)
            fusion_dim += item_id_dim
        else:
            self.item_embedding = None
            self.item_feature_dropout = nn.Identity()

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """ID embedding 小初始化，避免一开始压过内容模态。"""
        if self.item_embedding is not None:
            nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        text_features: Tensor,
        image_features: Tensor,
        video_features: Tensor,
        item_indices: Tensor | None = None,
    ) -> Tensor:
        if text_features.shape[:-1] != image_features.shape[:-1] or text_features.shape[:-1] != video_features.shape[:-1]:
            raise ValueError("text/image/video features must share the same leading shape.")
        if text_features.shape[-1] != self.text_dim:
            raise ValueError(f"text feature dim must be {self.text_dim}, got {text_features.shape[-1]}.")
        if image_features.shape[-1] != self.image_dim:
            raise ValueError(f"image feature dim must be {self.image_dim}, got {image_features.shape[-1]}.")
        if video_features.shape[-1] != self.video_dim:
            raise ValueError(f"video feature dim must be {self.video_dim}, got {video_features.shape[-1]}.")

        original_shape = text_features.shape[:-1]
        flat_text = text_features.reshape(-1, self.text_dim).float()
        flat_image = image_features.reshape(-1, self.image_dim).float()
        flat_video = video_features.reshape(-1, self.video_dim).float()

        features = [
            self.text_projection(flat_text),
            self.image_projection(flat_image),
            self.video_projection(flat_video),
        ]

        if self.use_item_id:
            if item_indices is None:
                raise ValueError("item_indices is required when use_item_id=True.")
            flat_item_indices = item_indices.reshape(-1).long().to(text_features.device)
            features.append(self.item_feature_dropout(self.item_embedding(flat_item_indices)))

        item_emb = self.fusion(torch.cat(features, dim=-1))
        if self.normalize_output:
            item_emb = F.normalize(item_emb, dim=-1)

        return item_emb.reshape(*original_shape, self.output_dim)

    def encode_item(
        self,
        text_features: Tensor,
        image_features: Tensor,
        video_features: Tensor,
        item_indices: Tensor | None = None,
    ) -> Tensor:
        """语义化别名，训练和评估统一调用 encode_item。"""
        return self.forward(
            text_features=text_features,
            image_features=image_features,
            video_features=video_features,
            item_indices=item_indices,
        )


def build_projection(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    """单模态投影层：降维、归一化、非线性。"""
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )
