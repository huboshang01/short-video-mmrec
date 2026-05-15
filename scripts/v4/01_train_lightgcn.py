"""
V4 Step 01：训练 MicroLens-100K LightGCN 协同过滤召回模型。

本脚本负责把 V3 已处理好的 MicroLens 行为样本转换为 user-item 二部图，
并训练一条纯协同过滤召回通道。与 V3 的多模态内容召回不同，LightGCN
只使用用户和短视频的交互关系，通过图传播学习 user/item embedding。

训练目标使用 BPR：
    对每条正反馈交互采样一个用户未交互负 item，
    让 score(user, positive_item) 高于 score(user, negative_item)。
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

from src.v4.data.microlens_graph_dataset import LightGCNInteractionDataset
from src.v4.models.lightgcn import LightGCN, build_normalized_adj
from src.v4.train.train_lightgcn import evaluate_sampled_bpr, save_lightgcn_checkpoint, train_one_epoch_bpr


# 默认配置与 checkpoint 文件名，保持训练输出目录结构稳定。
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v4" / "microlens_lightgcn.yaml"
TRAIN_CONFIG_NAME = "train_config.json"
LATEST_CHECKPOINT_NAME = "lightgcn_latest.pt"
BEST_CHECKPOINT_NAME = "lightgcn_best.pt"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，只暴露常用实验开关。"""
    parser = argparse.ArgumentParser(description="训练 V4 MicroLens LightGCN 召回模型。")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="V4 YAML/JSON 配置文件路径。")
    parser.add_argument("--output-dir", type=str, default="", help="覆盖配置中的输出目录。")
    parser.add_argument("--epochs", type=int, default=-1, help="大于 0 时覆盖训练轮数。")
    parser.add_argument("--batch-size", type=int, default=-1, help="大于 0 时覆盖 batch size。")
    parser.add_argument("--max-train-samples", type=int, default=None, help="覆盖最大训练样本数；-1 表示全量。")
    parser.add_argument("--max-val-samples", type=int, default=None, help="覆盖最大验证样本数；-1 表示全量。")
    parser.add_argument("--device", type=str, default="", help="覆盖运行设备，例如 cuda 或 cpu。")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """把项目内相对路径统一解析到仓库根目录。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    """checkpoint 中尽量保存相对路径，便于迁移项目目录。"""
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_config(path: str | Path) -> dict:
    """读取 YAML/JSON 配置文件。"""
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
    """将命令行参数覆盖到配置中，方便快速跑不同实验。"""
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
    """配置里 -1 表示使用全部样本；Dataset 里用 None 表示不限制。"""
    if value is None or int(value) == -1:
        return None
    if int(value) <= 0:
        raise ValueError("max samples must be positive or -1.")
    return int(value)


def set_seed(seed: int) -> None:
    """固定随机种子，让负采样和模型初始化尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """支持 auto/cuda/cpu，并在 CUDA 不可用时自动回退到 CPU。"""
    device_name = device_arg
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def build_dataloader(dataset: LightGCNInteractionDataset, config: dict, shuffle: bool) -> DataLoader:
    """把 BPR 三元组 Dataset 封装成 PyTorch DataLoader。"""
    train_cfg = config["train"]
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=False,
    )


def build_graph_tensors(dataset: LightGCNInteractionDataset, device: torch.device) -> torch.Tensor:
    """用 train 交互边构造 LightGCN 需要的归一化邻接矩阵。"""
    # user_index 和 item_index 是二部图的两侧节点，来自 train split 的正反馈边。
    user_indices = torch.tensor([edge.user_index for edge in dataset.interactions], dtype=torch.long)
    item_indices = torch.tensor([edge.item_index for edge in dataset.interactions], dtype=torch.long)
    # build_normalized_adj 会生成 D^-1/2 A D^-1/2，供 LightGCN 多层传播使用。
    return build_normalized_adj(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        user_indices=user_indices,
        item_indices=item_indices,
        device=device,
    )


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型总参数量，用于日志和 train_config 记录。"""
    return sum(param.numel() for param in model.parameters())


def build_train_config(
    config: dict,
    config_path: Path,
    train_dataset: LightGCNInteractionDataset,
    parameter_count: int,
) -> dict:
    """保存评估脚本复原模型所需的完整训练配置。"""
    return {
        "checkpoint_format": "v4_microlens_lightgcn_bpr",
        "config_path": project_relative(config_path),
        "config": config,
        "num_users": train_dataset.num_users,
        "num_items": train_dataset.num_items,
        "num_train_edges": len(train_dataset.interactions),
        "num_train_targets": len(train_dataset.targets),
        "parameter_count": {"total": parameter_count, "trainable": parameter_count},
    }


def save_train_config(output_dir: Path, train_config: dict) -> None:
    """单独写出 train_config.json，便于不加载 checkpoint 时查看实验配置。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / TRAIN_CONFIG_NAME).open("w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    # 1. 读取配置、应用命令行覆盖，并确定运行设备。
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = apply_cli_overrides(load_config(config_path), args)
    train_cfg = config["train"]
    set_seed(int(train_cfg["seed"]))
    device = resolve_device(str(train_cfg.get("device", "auto")))
    output_dir = resolve_project_path(config["output"]["dir"])

    # 2. 构造训练和验证 Dataset。
    # 训练集负责建立用户映射、item 映射、train_seen_items 和 BPR 正样本集合。
    train_dataset = LightGCNInteractionDataset(
        target_samples_path=resolve_project_path(config["data"]["train_samples"]),
        item_ids_path=resolve_project_path(config["data"]["item_ids"]),
        max_samples=normalize_max_samples(train_cfg.get("max_train_samples")),
        seed=int(train_cfg["seed"]),
    )
    # 验证集必须复用 train 的 user_id_to_index 和 user_seen_items，保证图节点编号一致，
    # 同时负采样避开 train 中已经交互过的 item。
    val_dataset = LightGCNInteractionDataset(
        target_samples_path=resolve_project_path(config["data"]["val_samples"]),
        item_ids_path=resolve_project_path(config["data"]["item_ids"]),
        user_id_to_index=train_dataset.user_id_to_index,
        user_seen_items=train_dataset.user_seen_items,
        max_samples=normalize_max_samples(train_cfg.get("max_val_samples")),
        seed=int(train_cfg["seed"]) + 10_000,
    )
    # DataLoader 只产出 user/positive_item/negative_item 三元组，不搬运大特征。
    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    val_loader = build_dataloader(val_dataset, config, shuffle=False)
    # LightGCN 的图结构固定来自 train split，训练和验证都使用同一张图。
    norm_adj = build_graph_tensors(train_dataset, device=device)

    # 3. 构造 LightGCN 模型。V4 只学习 ID embedding，不读取多模态内容特征。
    model_cfg = config["model"]
    model = LightGCN(
        num_users=train_dataset.num_users,
        num_items=train_dataset.num_items,
        embedding_dim=int(model_cfg["embedding_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        normalize_output=bool(model_cfg.get("normalize_output", False)),
    ).to(device)
    # AdamW 同时优化 user embedding 和 item embedding。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    # 4. 保存可复现实验所需的配置摘要。
    parameter_count = count_parameters(model)
    train_config = build_train_config(config, config_path, train_dataset, parameter_count)
    save_train_config(output_dir, train_config)

    print("=" * 80)
    print("V4 MicroLens LightGCN BPR Training")
    print("=" * 80)
    print(f"[INFO] config: {config_path}")
    print(f"[INFO] output dir: {output_dir}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] users: {train_dataset.num_users}")
    print(f"[INFO] items: {train_dataset.num_items}")
    print(f"[INFO] train graph edges: {len(train_dataset.interactions)}")
    print(f"[INFO] train target triples per epoch: {len(train_dataset)}")
    print(f"[INFO] val target triples: {len(val_dataset)}")
    print(f"[INFO] model params: {parameter_count:,}")
    print(model)

    best_val_loss = float("inf")
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        # 每轮更新 epoch，让 Dataset 的确定性负采样在不同 epoch 里变化。
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)

        print("=" * 80)
        print(f"Epoch {epoch}/{int(train_cfg['epochs'])}")
        # 训练阶段：当前 batch 的正负 item 通过 BPR loss 拉开分数。
        train_metrics = train_one_epoch_bpr(
            model=model,
            norm_adj=norm_adj,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            log_every=int(train_cfg["log_every"]),
        )
        # 验证阶段：仍使用 sampled negative，只评估 BPR 排序质量，不做 full-catalog。
        val_metrics = evaluate_sampled_bpr(
            model=model,
            norm_adj=norm_adj,
            dataloader=val_loader,
            device=device,
        )
        metrics = {"train": train_metrics, "val": val_metrics}
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        # latest 每轮覆盖，方便中断后拿到最近模型。
        save_lightgcn_checkpoint(
            output_dir / LATEST_CHECKPOINT_NAME,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=train_config,
        )
        # best 按验证 BPR loss 选择，供 full-catalog 评估脚本默认使用。
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            save_lightgcn_checkpoint(
                output_dir / BEST_CHECKPOINT_NAME,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=train_config,
            )

    print("=" * 80)
    print(f"Training done. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
