import torch.nn as nn
from torch import Tensor

from src.models.behavior_adapter import BehaviorAdapter


class ItemEncoder(nn.Module):
    """
    V2 item 编码器。

    当前流程：
        item_text
            -> BGE / sentence-transformers 向量缓存
            -> BehaviorAdapter 轻量适配
            -> 行为感知 item 向量
    说明：
        - 当前只使用 item_text_embedding 作为输入。
        - category_id 不作为输入特征，后续可作为辅助监督或 Category Hit@K 评估标签。
        - 这一层保留为 item 侧统一入口，方便后续替换/扩展 item encoder。
    """
    def __init__(
        self,
        text_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
        use_input_projection: bool = False,
        normalize_output: bool = True,
    ):
        super().__init__()
        if text_dim <= 0 or output_dim <= 0:
            raise ValueError("text_dim and output_dim must be positive.")

        self.text_dim = text_dim
        self.output_dim = output_dim

        # 用 BehaviorAdapter 将静态 BGE 文本向量适配到行为监督的召回空间
        self.adapter = BehaviorAdapter(
            input_dim=text_dim,
            output_dim=output_dim,
            dropout=dropout,
            use_input_projection=use_input_projection,
            normalize_output=normalize_output,
        )

    def forward(self, item_text_emb: Tensor) -> Tensor:
        """输入 BGE item 文本向量，输出行为感知 item 向量。"""
        return self.adapter.encode_item(item_text_emb)

    def encode_item(self, item_text_emb: Tensor) -> Tensor:
        """和 forward 等价，便于训练/评估代码语义化调用。"""
        return self.forward(item_text_emb)
