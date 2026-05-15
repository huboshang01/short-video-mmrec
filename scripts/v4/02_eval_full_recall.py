"""
V4 Step 02：对 MicroLens LightGCN checkpoint 做 full-catalog 召回评估。

评估流程与 V3 保持一致，但召回向量来自 LightGCN 协同过滤模型：
    1. 用 train split 重建 user-item 图；
    2. 通过 LightGCN 编码所有 user/item embedding；
    3. 对每个 eval 用户打分全量 item；
    4. 过滤 train 中已交互 item，避免推荐已看内容；
    5. 用 val/test 正反馈计算 Recall@K、HitRate@K、NDCG@K、MRR@K。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.eval.retrieval_metrics import aggregate_ranking_metrics
from src.v4.data.microlens_graph_dataset import (
    LightGCNInteractionDataset,
    build_eval_relevance,
    load_positive_rows,
)
from src.v4.models.lightgcn import LightGCN, build_normalized_adj


# 模块分工：
# - LightGCNInteractionDataset：复用 V3 CSV，重建 train 图和用户/item 映射。
# - LightGCN：根据 checkpoint 复原协同过滤向量模型。
# - aggregate_ranking_metrics：复用 V3 的 Recall/NDCG/MRR/HitRate 计算逻辑。

# 默认评估最新完整训练产物中的 best checkpoint。
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "v4" / "microlens_100k" / "lightgcn_bpr" / "lightgcn_best.pt"


def parse_args() -> argparse.Namespace:
    """解析评估脚本参数，支持快速 smoke test 和完整 test 评估。"""
    parser = argparse.ArgumentParser(description="评估 V4 MicroLens LightGCN full-catalog 召回效果。")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="V4 checkpoint 路径。")
    parser.add_argument("--eval-split", type=str, default="test", choices=["val", "test"], help="选择 val 或 test 评估。")
    parser.add_argument("--ks", type=str, default="", help="逗号分隔的 K 值；为空时使用 config eval.ks。")
    parser.add_argument("--output", type=str, default="", help="可选的 metrics JSON 输出路径。")
    parser.add_argument("--user-batch-size", type=int, default=-1, help="大于 0 时覆盖用户打分 batch size。")
    parser.add_argument("--max-eval-users", type=int, default=None, help="限制评估用户数用于快速验证；-1 表示全量。")
    parser.add_argument("--device", type=str, default="", help="覆盖运行设备，例如 cuda 或 cpu。")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """把项目内相对路径统一解析到仓库根目录。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(device_arg: str) -> torch.device:
    """支持 auto/cuda/cpu，并在 CUDA 不可用时自动回退到 CPU。"""
    device_name = device_arg
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def parse_ks(value: str, fallback: list[int]) -> list[int]:
    """解析命令行传入的 K 列表，默认使用训练配置中的 eval.ks。"""
    if not value:
        return sorted(set(int(k) for k in fallback))
    ks = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ks or min(ks) <= 0:
        raise ValueError("--ks must contain positive integers.")
    return sorted(set(ks))


def build_model_from_config(train_config: dict, device: torch.device) -> LightGCN:
    """根据 checkpoint 内保存的 train_config 复原 LightGCN 结构。"""
    config = train_config["config"]
    model_cfg = config["model"]
    model = LightGCN(
        num_users=int(train_config["num_users"]),
        num_items=int(train_config["num_items"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        normalize_output=bool(model_cfg.get("normalize_output", False)),
    )
    return model.to(device)


def build_graph_tensors(dataset: LightGCNInteractionDataset, device: torch.device) -> torch.Tensor:
    """用 train split 正反馈边重建 LightGCN 归一化邻接矩阵。"""
    # 评估必须和训练使用同一张 train 图，否则 user/item embedding 的传播口径会变。
    user_indices = torch.tensor([edge.user_index for edge in dataset.interactions], dtype=torch.long)
    item_indices = torch.tensor([edge.item_index for edge in dataset.interactions], dtype=torch.long)
    return build_normalized_adj(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        user_indices=user_indices,
        item_indices=item_indices,
        device=device,
    )


@torch.no_grad()
def rank_full_catalog(
    model: LightGCN,
    norm_adj: torch.Tensor,
    relevance: dict[int, set[int]],
    train_seen_items: dict[int, set[int]],
    user_batch_size: int,
    top_k: int,
    device: torch.device,
) -> dict[int, list[int]]:
    """批量为 eval 用户生成全量 item topK 排名。"""
    model.eval()
    # LightGCN 一次性编码全量 user/item embedding，后续只做矩阵乘法打分。
    user_embs, item_embs = model.encode_all(norm_adj)
    user_embs = user_embs.to(device)
    item_embs = item_embs.to(device)

    rankings: dict[int, list[int]] = {}
    user_indices = sorted(relevance.keys())
    for start in tqdm(range(0, len(user_indices), user_batch_size), desc="rank users"):
        # 取一批需要评估的用户，与全量 item embedding 做内积得到 [B, num_items] 分数矩阵。
        batch_users = user_indices[start:start + user_batch_size]
        batch_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
        scores = user_embs[batch_tensor] @ item_embs.T

        # 过滤 train 已交互 item，避免模型靠推荐用户已看内容获得不真实命中。
        for row_idx, user_index in enumerate(batch_users):
            seen = train_seen_items.get(int(user_index))
            if seen:
                scores[row_idx, list(seen)] = -float("inf")

        # 只保留最大 K 的排名结果，后续不同 K 的指标可从同一份 ranking 切片得到。
        k = min(top_k, scores.shape[1])
        top_indices = torch.topk(scores, k=k, dim=1).indices.detach().cpu().numpy()
        for user_index, ranking in zip(batch_users, top_indices, strict=False):
            rankings[int(user_index)] = [int(item_index) for item_index in ranking.tolist()]

    return rankings


def main() -> None:
    # 1. 加载 checkpoint，并从 checkpoint 中取回原始训练配置。
    args = parse_args()
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = checkpoint["config"]
    config = train_config["config"]

    # 2. 解析评估超参：K 列表、用户 batch size、是否限制评估用户数。
    device = resolve_device(args.device or str(config["train"].get("device", "auto")))
    ks = parse_ks(args.ks, fallback=list(config["eval"]["ks"]))
    user_batch_size = args.user_batch_size if args.user_batch_size > 0 else int(config["eval"]["user_batch_size"])
    max_eval_users_value = args.max_eval_users if args.max_eval_users is not None else int(config["eval"].get("max_eval_users", -1))
    max_eval_users = None if int(max_eval_users_value) == -1 else int(max_eval_users_value)

    # 3. 重建 train Dataset 和 train 图。
    # 这里 max_samples=None，确保评估使用完整 train 图，而不是训练 smoke 时的子样本图。
    train_dataset = LightGCNInteractionDataset(
        target_samples_path=resolve_project_path(config["data"]["train_samples"]),
        item_ids_path=resolve_project_path(config["data"]["item_ids"]),
        max_samples=None,
        seed=int(config["train"]["seed"]),
    )
    norm_adj = build_graph_tensors(train_dataset, device=device)

    # 4. 复原模型结构并加载权重。
    model = build_model_from_config(train_config, device)
    model.load_state_dict(checkpoint["model"])

    # 5. 聚合 val/test 正反馈作为相关 item 集合，指标以 user -> relevant_items 计算。
    eval_path = resolve_project_path(config["data"][f"{args.eval_split}_samples"])
    eval_rows = load_positive_rows(eval_path, train_dataset.item_id_to_index)
    relevance = build_eval_relevance(eval_rows, train_dataset.user_id_to_index)
    # 只评估 train 中有历史的用户，因为 LightGCN 对冷启动用户没有可传播的图表示。
    relevance = {user_index: items for user_index, items in relevance.items() if user_index in train_dataset.user_seen_items}
    if max_eval_users is not None:
        # 固定取排序后的前 N 个用户，保证 smoke test 可复现。
        keep_users = sorted(relevance.keys())[:max_eval_users]
        relevance = {user_index: relevance[user_index] for user_index in keep_users}

    print("=" * 80)
    print("V4 LightGCN Full-Catalog Retrieval Eval")
    print("=" * 80)
    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] eval split: {args.eval_split}")
    print(f"[INFO] eval samples: {eval_path}")
    print(f"[INFO] users with eval positives: {len(relevance)}")
    print(f"[INFO] users: {train_dataset.num_users}")
    print(f"[INFO] items: {train_dataset.num_items}")
    print(f"[INFO] ks: {ks}")

    # 6. 全量候选排序，并按 V3 复用的指标实现聚合 Recall/NDCG/MRR/HitRate。
    # rankings 的 key 是 user_index，value 是过滤已看 item 后的 topK item_index 列表。
    rankings = rank_full_catalog(
        model=model,
        norm_adj=norm_adj,
        relevance=relevance,
        train_seen_items=train_dataset.user_seen_items,
        user_batch_size=user_batch_size,
        top_k=max(ks),
        device=device,
    )
    metrics = aggregate_ranking_metrics(rankings, relevance, ks)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 7. 默认把指标写回 checkpoint 所在目录，便于一个实验目录里保存训练与评估产物。
    output_path = resolve_project_path(args.output) if args.output else checkpoint_path.parent / f"full_recall_{args.eval_split}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
