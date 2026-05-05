"""
V3 Step 02: 训练 MicroLens-100K 多模态双塔召回模型。

训练目标采用 sampled softmax：
    user(history) 与 1 个正样本 item、K 个采样负样本 item 做点积打分，
    CrossEntropy 要求正样本在候选集合中排第 1。

脚本职责：
    - 读取 configs/v3/microlens_mvp.yaml。
    - 构造 MicroLensRetrievalDataset 和 MultimodalFeatureCache。
    - 构造 MultimodalItemEncoder + RecentHistoryUserTower。
    - 训练并保存 latest/best checkpoint。
"""

from __future__ import annotations

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

from src.v3.data.microlens_retrieval_dataset import MicroLensRetrievalDataset, MultimodalFeatureCache
from src.v3.models.multimodal_item_encoder import MultimodalItemEncoder
from src.v3.models.user_tower import RecentHistoryUserTower
from src.v3.train.train_retrieval import (
    evaluate_sampled_softmax,
    save_retrieval_checkpoint,
    train_one_epoch_sampled_softmax,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v3" / "microlens_mvp.yaml"
TRAIN_CONFIG_NAME = "train_config.json"
LATEST_CHECKPOINT_NAME = "retrieval_latest.pt"
BEST_CHECKPOINT_NAME = "retrieval_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V3 MicroLens multimodal retrieval model.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to V3 YAML/JSON config.")
    parser.add_argument("--output-dir", type=str, default="", help="Override output directory.")
    parser.add_argument("--epochs", type=int, default=-1, help="Override epochs when > 0.")
    parser.add_argument("--batch-size", type=int, default=-1, help="Override batch size when > 0.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Override max train samples; -1 means all.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Override max val samples; -1 means all.")
    parser.add_argument("--device", type=str, default="", help="Override device, e.g. cuda or cpu.")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_config(path: str | Path) -> dict:
    """读取 YAML/JSON 配置；YAML 依赖 pyyaml，已在 requirements-v2 中列出。"""
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix.lower() == ".json":
            return json.load(f)
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Please install pyyaml or pass a JSON config file.") from exc
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """命令行参数只覆盖常用实验项，避免配置和脚本参数双重失控。"""
    if args.output_dir:
        config.setdefault("output", {})["dir"] = args.output_dir
    if args.epochs > 0:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size > 0:
        config.setdefault("train", {})["batch_size"] = args.batch_size
    if args.max_train_samples is not None:
        config.setdefault("train", {})["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        config.setdefault("train", {})["max_val_samples"] = args.max_val_samples
    if args.device:
        config.setdefault("train", {})["device"] = args.device
    return config


def normalize_max_samples(value: int | None) -> int | None:
    """配置中 -1 表示使用全部样本。"""
    if value is None or int(value) == -1:
        return None
    if int(value) <= 0:
        raise ValueError("max samples must be positive or -1.")
    return int(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """支持 config 中的 auto，并在 CUDA 不可用时回退到 CPU。"""
    device_name = device_arg
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def build_dataset(config: dict, split: str, max_samples: int | None) -> MicroLensRetrievalDataset:
    data_cfg = config["data"]
    train_samples = resolve_project_path(data_cfg["train_samples"])
    target_samples = resolve_project_path(data_cfg[f"{split}_samples"])

    # 验证集 target 用 val，但 history 只用 train，避免验证时偷看 val 行为。
    history_samples = train_samples
    return MicroLensRetrievalDataset(
        target_samples_path=target_samples,
        history_samples_path=history_samples,
        item_ids_path=resolve_project_path(data_cfg["item_ids"]),
        max_history_len=int(config["train"]["max_history_len"]),
        num_negatives=int(config["train"]["num_negatives"]),
        max_samples=max_samples,
        seed=int(config["train"]["seed"]) + (0 if split == "train" else 10_000),
    )


def build_dataloader(dataset: MicroLensRetrievalDataset, config: dict, shuffle: bool) -> DataLoader:
    train_cfg = config["train"]
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=False,
    )


def build_model(feature_cache: MultimodalFeatureCache, config: dict, device: torch.device):
    model_cfg = config["model"]
    item_tower = MultimodalItemEncoder(
        text_dim=feature_cache.text_dim,
        image_dim=feature_cache.image_dim,
        video_dim=feature_cache.video_dim,
        num_items=feature_cache.num_items,
        text_proj_dim=int(model_cfg["text_proj_dim"]),
        image_proj_dim=int(model_cfg["image_proj_dim"]),
        video_proj_dim=int(model_cfg["video_proj_dim"]),
        item_id_dim=int(model_cfg["item_id_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        dropout=float(model_cfg["dropout"]),
        id_feature_dropout=float(model_cfg["id_feature_dropout"]),
        use_item_id=bool(model_cfg["use_item_id"]),
        normalize_output=True,
    ).to(device)
    user_tower = RecentHistoryUserTower(
        input_dim=int(model_cfg["output_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        output_dim=int(model_cfg["output_dim"]),
        dropout=float(model_cfg["dropout"]),
        normalize_output=True,
    ).to(device)
    return item_tower, user_tower


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数量；trainable_only=True 时只计算参与训练的参数。"""
    params = model.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def build_parameter_summary(item_tower: torch.nn.Module, user_tower: torch.nn.Module) -> dict[str, dict[str, int] | int]:
    """记录 item/user tower 及整体参数量，方便不同 V3 版本横向比较。"""
    item_total = count_parameters(item_tower)
    user_total = count_parameters(user_tower)
    item_trainable = count_parameters(item_tower, trainable_only=True)
    user_trainable = count_parameters(user_tower, trainable_only=True)
    return {
        "item_tower": {"total": item_total, "trainable": item_trainable},
        "user_tower": {"total": user_total, "trainable": user_trainable},
        "total": item_total + user_total,
        "trainable": item_trainable + user_trainable,
    }


def print_model_summary(
    item_tower: torch.nn.Module,
    user_tower: torch.nn.Module,
    parameter_summary: dict[str, dict[str, int] | int],
) -> None:
    """打印模型结构和参数量，保持与 V2 训练脚本相近的启动信息。"""
    print("=" * 80)
    print("[MODEL]")
    print(item_tower)
    print(user_tower)
    print(
        f"item_tower params: trainable={parameter_summary['item_tower']['trainable']:,} | "
        f"total={parameter_summary['item_tower']['total']:,}"
    )
    print(
        f"user_tower params: trainable={parameter_summary['user_tower']['trainable']:,} | "
        f"total={parameter_summary['user_tower']['total']:,}"
    )
    print(
        f"all model params: trainable={parameter_summary['trainable']:,} | "
        f"total={parameter_summary['total']:,}"
    )


def build_train_config(
    config: dict,
    config_path: Path,
    feature_cache: MultimodalFeatureCache,
    parameter_summary: dict[str, dict[str, int] | int],
) -> dict:
    """checkpoint 内保存完整结构配置，评估脚本可直接复原模型。"""
    return {
        "checkpoint_format": "v3_microlens_sampled_softmax",
        "config_path": project_relative(config_path),
        "config": config,
        "num_items": feature_cache.num_items,
        "feature_dims": {
            "text": feature_cache.text_dim,
            "image": feature_cache.image_dim,
            "video": feature_cache.video_dim,
        },
        "parameter_count": parameter_summary,
    }


def save_train_config(output_dir: Path, train_config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / TRAIN_CONFIG_NAME).open("w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = apply_cli_overrides(load_config(config_path), args)
    train_cfg = config["train"]
    set_seed(int(train_cfg["seed"]))

    device = resolve_device(str(train_cfg.get("device", "auto")))

    output_dir = resolve_project_path(config["output"]["dir"])
    feature_cache = MultimodalFeatureCache.from_config(resolve_project_path(config["data"]["feature_config"]))
    feature_cache.validate()

    print("=" * 80)
    print("V3 MicroLens Multimodal Retrieval Training")
    print("=" * 80)
    print(f"[INFO] config: {config_path}")
    print(f"[INFO] output dir: {output_dir}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] num_items: {feature_cache.num_items}")
    print(f"[INFO] feature dims: text={feature_cache.text_dim}, image={feature_cache.image_dim}, video={feature_cache.video_dim}")

    train_dataset = build_dataset(config, split="train", max_samples=normalize_max_samples(train_cfg.get("max_train_samples")))
    val_dataset = build_dataset(config, split="val", max_samples=normalize_max_samples(train_cfg.get("max_val_samples")))
    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    val_loader = build_dataloader(val_dataset, config, shuffle=False)
    print(f"[INFO] train samples with history: {len(train_dataset)}")
    print(f"[INFO] val samples with train history: {len(val_dataset)}")

    item_tower, user_tower = build_model(feature_cache, config, device)
    optimizer = torch.optim.AdamW(
        list(item_tower.parameters()) + list(user_tower.parameters()),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    parameter_summary = build_parameter_summary(item_tower, user_tower)
    print_model_summary(item_tower, user_tower, parameter_summary)

    train_config = build_train_config(config, config_path, feature_cache, parameter_summary)
    save_train_config(output_dir, train_config)

    best_val_loss = float("inf")
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        print("=" * 80)
        print(f"Epoch {epoch}/{int(train_cfg['epochs'])}")
        train_metrics = train_one_epoch_sampled_softmax(
            item_tower=item_tower,
            user_tower=user_tower,
            feature_cache=feature_cache,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=float(train_cfg["temperature"]),
            log_every=int(train_cfg["log_every"]),
        )
        val_metrics = evaluate_sampled_softmax(
            item_tower=item_tower,
            user_tower=user_tower,
            feature_cache=feature_cache,
            dataloader=val_loader,
            device=device,
            temperature=float(train_cfg["temperature"]),
        )
        metrics = {"train": train_metrics, "val": val_metrics}
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        save_retrieval_checkpoint(
            output_dir / LATEST_CHECKPOINT_NAME,
            item_tower,
            user_tower,
            optimizer,
            epoch,
            metrics,
            train_config,
        )
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            save_retrieval_checkpoint(
                output_dir / BEST_CHECKPOINT_NAME,
                item_tower,
                user_tower,
                optimizer,
                epoch,
                metrics,
                train_config,
            )

    print("=" * 80)
    print(f"Training done. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
