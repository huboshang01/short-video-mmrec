"""
V2 双塔 InfoNCE 训练入口。

脚本职责：
    1. 读取 Step 02 生成的 item 文本向量缓存配置。
    2. 构造 TwoTowerBehaviorDataset 和 DataLoader。
    3. 初始化共享 ItemEncoder、UserEncoder、UserTower。
    4. 调用 src.train.train_v2 中的训练/验证循环，并保存 checkpoint。

注意：
    这里不重新定义正样本规则；正样本由 behavior_samples_*.csv 中的 is_positive 字段决定。
    训练和验证默认都只把 is_positive == 1 的行为样本作为 target item。
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.two_tower_dataset import TwoTowerBehaviorDataset
from src.models.behavior_adapter import count_trainable_parameters
from src.models.item_encoder import ItemEncoder
from src.models.user_encoder import UserEncoder, UserTower
from src.train.train_v2 import evaluate_inbatch, save_checkpoint, train_one_epoch


LINE_WIDTH = 80
TRAIN_CONFIG_NAME = "two_tower_train_config.json"
LATEST_CHECKPOINT_NAME = "two_tower_infonce_latest.pt"
BEST_CHECKPOINT_NAME = "two_tower_infonce_best.pt"


@dataclass(frozen=True)
class EmbeddingResources:
    """Step 02 item 文本向量缓存及其元信息。"""

    config_path: Path
    config: dict
    item_text_embeddings_path: Path
    item_ids_path: Path
    text_dim: int
    output_dim: int


# ---------------------------------------------------------------------------
# 命令行参数模块：集中声明训练入口需要的路径、采样、模型和优化参数。
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V2 two-tower InfoNCE model.")

    parser.add_argument("--embedding-config", type=str, default="outputs/v2/embeddings/embedding_config.json", help="Path to Step 02 embedding_config.json.")
    parser.add_argument("--train-samples", type=str, default="data/processed/v2/behavior_samples_train.csv", help="Path to behavior_samples_train.csv.")
    parser.add_argument("--val-samples", type=str, default="data/processed/v2/behavior_samples_val.csv", help="Path to behavior_samples_val.csv.")
    parser.add_argument("--output-dir", type=str, default="outputs/v2/checkpoints", help="Directory for train config and checkpoints.")

    parser.add_argument("--max-history-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--symmetric-loss", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--max-train-samples", type=int, default=200000, help="Limit training target samples for local experiments. Use -1 for all.")
    parser.add_argument("--max-val-samples", type=int, default=50000, help="Limit validation target samples for local experiments. Use -1 for all.")
    parser.add_argument("--history-min-watch-ratio", type=float, default=0.0, help="Minimum watch_ratio used when building user histories.")
    parser.add_argument("--use-input-projection", action="store_true", help="Use a Linear projection before the residual projection head.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device, for example cuda or cpu.")

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """在真正读大文件前做基础参数检查，避免跑到中途才失败。"""
    if args.max_history_len <= 0:
        raise ValueError("--max-history-len must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if args.history_min_watch_ratio < 0:
        raise ValueError("--history-min-watch-ratio must be non-negative.")
    if args.max_train_samples == 0 or args.max_train_samples < -1:
        raise ValueError("--max-train-samples must be positive or -1.")
    if args.max_val_samples == 0 or args.max_val_samples < -1:
        raise ValueError("--max-val-samples must be positive or -1.")


# ---------------------------------------------------------------------------
# 通用工具模块：路径解析、随机种子和 JSON 读取。
# ---------------------------------------------------------------------------
def resolve_project_path(path: str | Path) -> Path:
    """支持绝对路径，也支持相对项目根目录的路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    """固定 Python、NumPy、PyTorch 随机种子，保证采样和初始化尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str | Path) -> dict:
    """读取 UTF-8 JSON 配置文件。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_max_samples(value: int) -> int | None:
    """命令行中 -1 表示使用全部 target 样本，传给 Dataset 时转成 None。"""
    return None if value == -1 else value


# ---------------------------------------------------------------------------
# embedding 资源模块：把 embedding_config.json 解析成 Dataset 可直接使用的路径。
# ---------------------------------------------------------------------------
def load_embedding_resources(embedding_config_arg: str | Path) -> EmbeddingResources:
    embedding_config_path = resolve_project_path(embedding_config_arg)
    embedding_config = load_json(embedding_config_path)

    required_keys = {"embedding_path", "item_ids_path", "embedding_dim"}
    missing_keys = required_keys - set(embedding_config)
    if missing_keys:
        raise ValueError(f"Missing keys in embedding config: {missing_keys}")

    item_text_embeddings_path = resolve_project_path(embedding_config["embedding_path"])
    item_ids_path = resolve_project_path(embedding_config["item_ids_path"])
    text_dim = int(embedding_config["embedding_dim"])

    # 当前 ItemEncoder 不改变维度，output_dim 与 BGE text_dim 保持一致。
    output_dim = text_dim

    return EmbeddingResources(
        config_path=embedding_config_path,
        config=embedding_config,
        item_text_embeddings_path=item_text_embeddings_path,
        item_ids_path=item_ids_path,
        text_dim=text_dim,
        output_dim=output_dim,
    )


# ---------------------------------------------------------------------------
# 数据模块：构造 train/val Dataset 和 DataLoader。
# ---------------------------------------------------------------------------
def build_two_tower_dataset(
    behavior_samples_path: str | Path,
    resources: EmbeddingResources,
    args: argparse.Namespace,
    max_samples: int | None,
) -> TwoTowerBehaviorDataset:
    """构造双塔样本；target 默认只来自 is_positive == 1 的行为行。"""
    return TwoTowerBehaviorDataset(
        behavior_samples_path=resolve_project_path(behavior_samples_path),
        item_embeddings_path=resources.item_text_embeddings_path,
        item_ids_path=resources.item_ids_path,
        max_history_len=args.max_history_len,
        only_positive_targets=True,
        history_min_watch_ratio=args.history_min_watch_ratio,
        max_samples=max_samples,
        seed=args.seed,
    )


def build_dataloader(
    dataset: TwoTowerBehaviorDataset,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    """把 Dataset 组装成固定 batch；drop_last=True 保持 InfoNCE batch 规模稳定。"""
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=True,
    )


def build_data_pipeline(
    args: argparse.Namespace,
    resources: EmbeddingResources,
) -> tuple[TwoTowerBehaviorDataset, TwoTowerBehaviorDataset, DataLoader, DataLoader]:
    max_train_samples = normalize_max_samples(args.max_train_samples)
    max_val_samples = normalize_max_samples(args.max_val_samples)

    train_dataset = build_two_tower_dataset(
        behavior_samples_path=args.train_samples,
        resources=resources,
        args=args,
        max_samples=max_train_samples,
    )
    val_dataset = build_two_tower_dataset(
        behavior_samples_path=args.val_samples,
        resources=resources,
        args=args,
        max_samples=max_val_samples,
    )

    train_loader = build_dataloader(train_dataset, args, shuffle=True)
    val_loader = build_dataloader(val_dataset, args, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader


# ---------------------------------------------------------------------------
# 模型模块：初始化共享 ItemEncoder、UserEncoder 和 UserTower。
# ---------------------------------------------------------------------------
def build_model(
    resources: EmbeddingResources,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ItemEncoder, UserEncoder, UserTower]:
    item_encoder = ItemEncoder(
        text_dim=resources.text_dim,
        output_dim=resources.output_dim,
        dropout=args.dropout,
        use_input_projection=args.use_input_projection,
        normalize_output=True,
    ).to(device)

    user_encoder = UserEncoder(
        input_dim=resources.output_dim,
        output_dim=resources.output_dim,
        dropout=args.dropout,
        use_input_projection=args.use_input_projection,
        normalize_output=True,
    ).to(device)

    # UserTower 持有同一个 item_encoder；历史 item 和 target item 会进入同一行为语义空间。
    user_tower = UserTower(item_encoder=item_encoder, user_encoder=user_encoder).to(device)

    return item_encoder, user_encoder, user_tower


def build_optimizer(user_tower: UserTower, args: argparse.Namespace) -> torch.optim.Optimizer:
    """优化 UserTower 下挂载的全部可训练参数，包括共享 ItemEncoder 和 UserEncoder。"""
    return torch.optim.AdamW(
        user_tower.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


# ---------------------------------------------------------------------------
# 日志与保存模块：集中打印训练上下文、保存配置和 checkpoint。
# ---------------------------------------------------------------------------
def print_run_summary(
    args: argparse.Namespace,
    resources: EmbeddingResources,
    device: torch.device,
) -> None:
    max_train_samples = normalize_max_samples(args.max_train_samples)
    max_val_samples = normalize_max_samples(args.max_val_samples)

    print("=" * LINE_WIDTH)
    print("V2 Two-Tower InfoNCE Training")
    print("=" * LINE_WIDTH)
    print(f"[INFO] project root: {PROJECT_ROOT}")
    print(f"[INFO] embedding config: {resources.config_path}")
    print(f"[INFO] item text embeddings: {resources.item_text_embeddings_path}")
    print(f"[INFO] item ids: {resources.item_ids_path}")
    print(f"[INFO] text_dim: {resources.text_dim}")
    print(f"[INFO] output_dim: {resources.output_dim}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] max_history_len: {args.max_history_len}")
    print(f"[INFO] batch_size: {args.batch_size}")
    print(f"[INFO] max_train_samples: {max_train_samples}")
    print(f"[INFO] max_val_samples: {max_val_samples}")


def print_data_summary(
    train_dataset: TwoTowerBehaviorDataset,
    val_dataset: TwoTowerBehaviorDataset,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> None:
    print(f"[INFO] train target samples: {len(train_dataset)}")
    print(f"[INFO] val target samples: {len(val_dataset)}")
    print(f"[INFO] train batches: {len(train_loader)}")
    print(f"[INFO] val batches: {len(val_loader)}")


def print_model_summary(
    item_encoder: ItemEncoder,
    user_encoder: UserEncoder,
    user_tower: UserTower,
) -> None:
    print("\n[Model]")
    print(item_encoder)
    print(user_encoder)

    print("\n[Parameters]")
    print(f"item encoder trainable params: {count_trainable_parameters(item_encoder)}")
    print(f"user encoder trainable params: {count_trainable_parameters(user_encoder)}")
    print(f"total trainable params: {count_trainable_parameters(user_tower)}")


def build_train_config(args: argparse.Namespace, resources: EmbeddingResources) -> dict:
    """保存完整训练上下文，方便 checkpoint 之后复现实验。"""
    train_config = vars(args).copy()
    train_config.update(
        {
            "text_dim": resources.text_dim,
            "output_dim": resources.output_dim,
            "embedding_config_path": str(resources.config_path),
            "item_text_embeddings_path": str(resources.item_text_embeddings_path),
            "item_ids_path": str(resources.item_ids_path),
            "embedding_config": resources.config,
        }
    )
    return train_config


def save_train_config(output_dir: Path, train_config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / TRAIN_CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)


def save_epoch_checkpoints(
    output_dir: Path,
    item_encoder: ItemEncoder,
    user_encoder: UserEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    train_config: dict,
    best_val_loss: float,
) -> float:
    """每个 epoch 保存 latest；验证 loss 创新低时额外保存 best。"""
    checkpoint_metrics = {"train": train_metrics, "val": val_metrics}

    latest_path = output_dir / LATEST_CHECKPOINT_NAME
    save_checkpoint(
        path=latest_path,
        item_encoder=item_encoder,
        user_encoder=user_encoder,
        optimizer=optimizer,
        epoch=epoch,
        metrics=checkpoint_metrics,
        config=train_config,
    )

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        best_path = output_dir / BEST_CHECKPOINT_NAME
        save_checkpoint(
            path=best_path,
            item_encoder=item_encoder,
            user_encoder=user_encoder,
            optimizer=optimizer,
            epoch=epoch,
            metrics=checkpoint_metrics,
            config=train_config,
        )
        print(f"[Saved] best checkpoint: {best_path}")

    return best_val_loss


# ---------------------------------------------------------------------------
# 训练主流程模块：串起配置、数据、模型、优化器和 epoch 循环。
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    resources = load_embedding_resources(args.embedding_config)
    output_dir = resolve_project_path(args.output_dir)

    print_run_summary(args=args, resources=resources, device=device)

    train_dataset, val_dataset, train_loader, val_loader = build_data_pipeline(
        args=args,
        resources=resources,
    )
    print_data_summary(train_dataset, val_dataset, train_loader, val_loader)

    item_encoder, user_encoder, user_tower = build_model(
        resources=resources,
        args=args,
        device=device,
    )
    print_model_summary(item_encoder, user_encoder, user_tower)

    optimizer = build_optimizer(user_tower=user_tower, args=args)

    train_config = build_train_config(args=args, resources=resources)
    save_train_config(output_dir=output_dir, train_config=train_config)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * LINE_WIDTH)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * LINE_WIDTH)

        train_metrics = train_one_epoch(
            item_encoder=item_encoder,
            user_tower=user_tower,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
            symmetric_loss=args.symmetric_loss,
            log_every=args.log_every,
        )

        val_metrics = evaluate_inbatch(
            item_encoder=item_encoder,
            user_tower=user_tower,
            dataloader=val_loader,
            device=device,
            temperature=args.temperature,
            symmetric_loss=args.symmetric_loss,
        )

        print(f"\n[Epoch {epoch}] train: {train_metrics}")
        print(f"[Epoch {epoch}] val:   {val_metrics}")

        best_val_loss = save_epoch_checkpoints(
            output_dir=output_dir,
            item_encoder=item_encoder,
            user_encoder=user_encoder,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            train_config=train_config,
            best_val_loss=best_val_loss,
        )

    print("\n[Done] Two-tower InfoNCE training finished.")


if __name__ == "__main__":
    main()
