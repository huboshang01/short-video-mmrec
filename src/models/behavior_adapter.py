import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualAdd(nn.Module):
    """残差包装层：输出 x + fn(x)，和 NICE++ 中的残差投影头保持一致。"""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class ProjectionHead(nn.Module):
    """
    残差 MLP 投影头。

    NICE++ 中的用途：
        EEG / image feature -> ProjectionHead -> CLIP 对齐空间

    本项目 V2 中的用途：
        BGE item text embedding -> ProjectionHead -> 行为感知 item 语义空间
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        use_input_projection: bool = True,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 当输入/输出维度不同，或显式要求投影时，先做线性维度变换
        self.input_projection = (
            nn.Linear(input_dim, output_dim)
            if use_input_projection or input_dim != output_dim
            else nn.Identity()
        )

        # 残差 MLP 负责在原始语义空间基础上学习一层行为偏好修正
        self.residual = ResidualAdd(
            nn.Sequential(
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.Dropout(dropout),
            )
        )

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x.float())
        x = self.residual(x)
        return self.norm(x)


class BehaviorAdapter(nn.Module):
    """
    行为感知 Item Adapter。

    输入：
        item_text_emb: BGE / sentence-transformers 生成的 item 文本向量

    输出：
        item_emb: 经过行为监督适配后的 item 向量

    说明：
        - category_id 不作为 adapter 输入，只作为后续辅助监督或评估标签。
        - normalize_output=True 时，输出可直接用于 cosine / InfoNCE / FAISS IP 检索。
        - 结构上呼应 NICE++ 的 projection head，只是这里对齐的是“内容语义”和“用户行为偏好”。
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
        use_input_projection: bool = False,
        normalize_output: bool = True,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")

        self.projection = ProjectionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
            use_input_projection=use_input_projection,
        )
        self.normalize_output = normalize_output

    def forward(self, item_text_emb: Tensor) -> Tensor:
        """将 BGE item 文本向量映射到行为感知召回空间。"""
        item_emb = self.projection(item_text_emb)

        if self.normalize_output:
            item_emb = F.normalize(item_emb, dim=-1)

        return item_emb

    def encode_item(self, item_text_emb: Tensor) -> Tensor:
        """语义更明确的 forward 别名，方便训练和评估代码调用。"""
        return self.forward(item_text_emb)


def count_trainable_parameters(model: nn.Module) -> int:
    """统计模型中需要训练的参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
