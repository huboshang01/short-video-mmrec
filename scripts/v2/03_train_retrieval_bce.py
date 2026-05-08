"""
V2.1 时间感知双塔召回训练入口。

本脚本是旧 V2 InfoNCE 训练的 MVP 替代版本：
    1. Dataset 使用 TimeAwarePointwiseRetrievalDataset，保证 history 来自 target 之前。
    2. Loss 使用 BCE，显式学习正反馈和负反馈。
    3. ItemTower 使用 text/item_id/category concat 融合，兼顾语义和协同信号。
    4. UserTower 使用最近历史 + watch_ratio + time decay 构造用户兴趣向量。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2.data.time_aware_retrieval_dataset import TimeAwarePointwiseRetrievalDataset, build_category_vocab
from src.v2.models.retrieval_item_tower import TextIdCategoryItemTower
from src.v2.models.retrieval_user_tower import RecentHistoryUserTower
from src.v2.train.train_retrieval import evaluate_pointwise_bce, save_retrieval_checkpoint, train_one_epoch_bce


LINE_WIDTH = 80
TRAIN_CONFIG_NAME = "retrieval_bce_train_config.json"
LATEST_CHECKPOINT_NAME = "retrieval_bce_latest.pt"
BEST_CHECKPOINT_NAME = "retrieval_bce_best.pt"


@dataclass(frozen=True)
class EmbeddingResources:
    """embedding_config.json 解析后的 item 文本向量资源。"""

    config_path: Path
    config: dict
    item_text_embeddings_path: Path
    item_ids_path: Path
    text_dim: int
    num_items: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V2.1 time-aware pointwise retrieval model.")
    parser.add_argument("--embedding-config", type=str, default="outputs/v2/embeddings/embedding_config.json", help="Path to Step 02 embedding_config.json.")
    parser.add_argument("--train-samples", type=str, default="data/processed/v2/behavior_samples_train.csv", help="Path to time-aware train samples.")
    parser.add_argument("--val-samples", type=str, default="data/processed/v2/behavior_samples_val.csv", help="Path to time-aware val samples.")
    parser.add_argument("--output-dir", type=str, default="outputs/v2/retrieval_bce", help="Directory for train config and checkpoints.")
    parser.add_argument("--max-history-len", type=int, default=50)
    parser.add_argument("--history-min-watch-ratio", type=float, default=1.0, help="Only previous interactions with watch_ratio >= this value enter user history.")
    parser.add_argument("--pos-threshold", type=float, default=1.0)
    parser.add_argument("--neg-threshold", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--text-proj-dim", type=int, default=256)
    parser.add_argument("--item-id-dim", type=int, default=64)
    parser.add_argument("--category-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--id-feature-dropout", type=float, default=0.1)
    parser.add_argument("--watch-weight-cap", type=float, default=5.0)
    parser.add_argument("--time-decay-days", type=float, default=7.0)
    parser.add_argument("--no-item-id", action="store_true", help="Disable item_id embedding for ablation.")
    parser.add_argument("--no-category", action="store_true", help="Disable category embedding for ablation.")
    parser.add_argument("--max-train-samples", type=int, default=200000, help="Limit train target samples for fast experiments. Use -1 for all.")
    parser.add_argument("--max-val-samples", type=int, default=50000, help="Limit val target samples for fast experiments. Use -1 for all.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """在读大文件和初始化模型前做参数检查。"""
    positive_fields = [
        ("max_history_len", args.max_history_len),
        ("batch_size", args.batch_size),
        ("epochs", args.epochs),
        ("lr", args.lr),
        ("temperature", args.temperature),
        ("text_proj_dim", args.text_proj_dim),
        ("item_id_dim", args.item_id_dim),
        ("category_dim", args.category_dim),
        ("hidden_dim", args.hidden_dim),
        ("output_dim", args.output_dim),
        ("watch_weight_cap", args.watch_weight_cap),
        ("time_decay_days", args.time_decay_days),
        ("log_every", args.log_every),
    ]
    for name, value in positive_fields:
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.neg_threshold >= args.pos_threshold:
        raise ValueError("--neg-threshold must be smaller than --pos-threshold.")
    if args.history_min_watch_ratio < 0:
        raise ValueError("--history-min-watch-ratio must be non-negative.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be non-negative.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")
    if not 0 <= args.id_feature_dropout < 1:
        raise ValueError("--id-feature-dropout must be in [0, 1).")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.max_train_samples == 0 or args.max_train_samples < -1:
        raise ValueError("--max-train-samples must be positive or -1.")
    if args.max_val_samples == 0 or args.max_val_samples < -1:
        raise ValueError("--max-val-samples must be positive or -1.")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_max_samples(value: int) -> int | None:
    """命令行中 -1 表示使用全部样本。"""
    return None if value == -1 else int(value)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embedding_resources(embedding_config_arg: str | Path) -> EmbeddingResources:
    embedding_config_path = resolve_project_path(embedding_config_arg)
    embedding_config = load_json(embedding_config_path)

    required_keys = {"embedding_path", "item_ids_path", "embedding_dim"}
    missing_keys = required_keys - set(embedding_config)
    if missing_keys:
        raise ValueError(f"Missing keys in embedding config: {missing_keys}")

    item_text_embeddings_path = resolve_project_path(embedding_config["embedding_path"])
    item_ids_path = resolve_project_path(embedding_config["item_ids_path"])
    item_ids = np.load(item_ids_path).astype("int64")

    return EmbeddingResources(
        config_path=embedding_config_path,
        config=embedding_config,
        item_text_embeddings_path=item_text_embeddings_path,
        item_ids_path=item_ids_path,
        text_dim=int(embedding_config["embedding_dim"]),
        num_items=int(len(item_ids)),
    )


def build_dataset(
    samples_path: str | Path,
    resources: EmbeddingResources,
    category_to_index: dict[str, int],
    args: argparse.Namespace,
    max_samples: int | None,
) -> TimeAwarePointwiseRetrievalDataset:
    return TimeAwarePointwiseRetrievalDataset(
        behavior_samples_path=resolve_project_path(samples_path),
        item_embeddings_path=resources.item_text_embeddings_path,
        item_ids_path=resources.item_ids_path,
        category_to_index=category_to_index,
        max_history_len=args.max_history_len,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        history_min_watch_ratio=args.history_min_watch_ratio,
        max_samples=max_samples,
        seed=args.seed,
    )


def build_dataloader(dataset: TimeAwarePointwiseRetrievalDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=False)


def build_model(
    resources: EmbeddingResources,
    category_to_index: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TextIdCategoryItemTower, RecentHistoryUserTower]:
    item_tower = TextIdCategoryItemTower(
        text_dim=resources.text_dim,
        num_items=resources.num_items,
        num_categories=len(category_to_index),
        text_proj_dim=args.text_proj_dim,
        item_id_dim=args.item_id_dim,
        category_dim=args.category_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        id_feature_dropout=args.id_feature_dropout,
        normalize_output=True,
        use_item_id=not args.no_item_id,
        use_category=not args.no_category,
    ).to(device)

    user_tower = RecentHistoryUserTower(
        item_tower=item_tower,
        input_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        watch_weight_cap=args.watch_weight_cap,
        time_decay_days=args.time_decay_days,
        normalize_output=True,
    ).to(device)

    return item_tower, user_tower


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_train_config(args: argparse.Namespace, resources: EmbeddingResources, category_to_index: dict[str, int]) -> dict:
    return {
        "args": vars(args),
        "embedding_config_path": str(resources.config_path),
        "item_text_embeddings_path": str(resources.item_text_embeddings_path),
        "item_ids_path": str(resources.item_ids_path),
        "embedding_config": resources.config,
        "num_items": resources.num_items,
        "text_dim": resources.text_dim,
        "category_to_index": category_to_index,
        "checkpoint_format": "v2.1_retrieval_bce",
    }


def save_train_config(output_dir: Path, train_config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / TRAIN_CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)


def save_epoch_checkpoints(
    output_dir: Path,
    item_tower: TextIdCategoryItemTower,
    user_tower: RecentHistoryUserTower,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    train_config: dict,
    best_val_loss: float,
) -> float:
    metrics = {"train": train_metrics, "val": val_metrics}
    latest_path = output_dir / LATEST_CHECKPOINT_NAME
    save_retrieval_checkpoint(latest_path, item_tower, user_tower, optimizer, epoch, metrics, train_config)

    val_loss = float(val_metrics["loss"])
    if val_loss < best_val_loss:
        best_path = output_dir / BEST_CHECKPOINT_NAME
        save_retrieval_checkpoint(best_path, item_tower, user_tower, optimizer, epoch, metrics, train_config)
        best_val_loss = val_loss

    return best_val_loss


def print_run_summary(args: argparse.Namespace, resources: EmbeddingResources, category_to_index: dict[str, int], device: torch.device) -> None:
    print("=" * LINE_WIDTH)
    print("V2.1 Time-Aware Pointwise Retrieval Training")
    print("=" * LINE_WIDTH)
    print(f"[INFO] device: {device}")
    print(f"[INFO] embedding config: {resources.config_path}")
    print(f"[INFO] item text embeddings: {resources.item_text_embeddings_path}")
    print(f"[INFO] train samples: {resolve_project_path(args.train_samples)}")
    print(f"[INFO] val samples: {resolve_project_path(args.val_samples)}")
    print(f"[INFO] num_items: {resources.num_items}")
    print(f"[INFO] num_categories: {len(category_to_index)}")
    print(f"[INFO] max_history_len: {args.max_history_len}")
    print(f"[INFO] history_min_watch_ratio: {args.history_min_watch_ratio}")
    print(f"[INFO] batch_size: {args.batch_size}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    resources = load_embedding_resources(args.embedding_config)
    output_dir = resolve_project_path(args.output_dir)

    category_to_index = build_category_vocab([resolve_project_path(args.train_samples), resolve_project_path(args.val_samples)])
    print_run_summary(args, resources, category_to_index, device)

    train_dataset = build_dataset(args.train_samples, resources, category_to_index, args, normalize_max_samples(args.max_train_samples))
    val_dataset = build_dataset(args.val_samples, resources, category_to_index, args, normalize_max_samples(args.max_val_samples))
    train_loader = build_dataloader(train_dataset, args, shuffle=True)
    val_loader = build_dataloader(val_dataset, args, shuffle=False)

    print(f"[INFO] train target samples: {len(train_dataset)} | positive_ratio={float(train_dataset.labels.mean()):.4f}")
    print(f"[INFO] val target samples: {len(val_dataset)} | positive_ratio={float(val_dataset.labels.mean()):.4f}")
    print(f"[INFO] train batches: {len(train_loader)}")
    print(f"[INFO] val batches: {len(val_loader)}")

    item_tower, user_tower = build_model(resources, category_to_index, args, device)
    print("=" * LINE_WIDTH)
    print("[MODEL]")
    print(item_tower)
    print(user_tower)
    print(f"trainable params: {count_trainable_parameters(user_tower)}")

    optimizer = torch.optim.AdamW(user_tower.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_config = build_train_config(args, resources, category_to_index)
    save_train_config(output_dir, train_config)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print("\n" + "=" * LINE_WIDTH)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * LINE_WIDTH)

        train_metrics = train_one_epoch_bce(
            item_tower=item_tower,
            user_tower=user_tower,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
            log_every=args.log_every,
        )
        val_metrics = evaluate_pointwise_bce(
            item_tower=item_tower,
            user_tower=user_tower,
            dataloader=val_loader,
            device=device,
            temperature=args.temperature,
        )

        print(f"\n[Epoch {epoch}] train: {train_metrics}")
        print(f"[Epoch {epoch}] val:   {val_metrics}")

        best_val_loss = save_epoch_checkpoints(
            output_dir=output_dir,
            item_tower=item_tower,
            user_tower=user_tower,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            train_config=train_config,
            best_val_loss=best_val_loss,
        )

    print("\n[Done] V2.1 retrieval BCE training finished.")


if __name__ == "__main__":
    main()
