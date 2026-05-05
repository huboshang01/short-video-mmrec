"""
V2.1 全量 item 召回评估。

评估口径采用 MVP 默认的 fixed_history：
    - user embedding 只由 train split 中的历史正反馈构造。
    - 在全量 item 库上打分召回。
    - 过滤 train 中已正反馈 item。
    - 用 val/test 的正反馈 item 计算 Recall/NDCG/MRR/HitRate。

这比旧版 in-batch Acc 更接近真实召回系统：线上也是先固定用户画像，再从全量候选库取 topK。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.retrieval_metrics import aggregate_ranking_metrics
from src.models.retrieval_item_tower import TextIdCategoryItemTower
from src.models.retrieval_user_tower import RecentHistoryUserTower


UNKNOWN_CATEGORY = "__UNK__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V2.1 retrieval checkpoint with full-catalog ranking.")
    parser.add_argument("--checkpoint", type=str, default="outputs/v2/retrieval_bce/retrieval_bce_best.pt", help="Path to retrieval BCE checkpoint.")
    parser.add_argument("--train-samples", type=str, default="data/processed/v2/behavior_samples_train.csv", help="Train samples used as fixed user history.")
    parser.add_argument("--eval-samples", type=str, default="data/processed/v2/behavior_samples_val.csv", help="Val/test samples containing positive targets.")
    parser.add_argument("--ks", type=str, default="10,20,50,100", help="Comma-separated K values.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--user-batch-size", type=int, default=256)
    parser.add_argument("--history-min-watch-ratio", type=float, default=-1.0, help="Override checkpoint history threshold when >= 0.")
    parser.add_argument("--output", type=str, default="", help="Optional JSON metrics output path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.user_batch_size <= 0:
        raise ValueError("batch sizes must be positive.")
    return args


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_ks(value: str) -> list[int]:
    ks = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ks or min(ks) <= 0:
        raise ValueError("--ks must contain positive integers.")
    return sorted(set(ks))


def load_behavior_samples(path: Path, pos_threshold: float, neg_threshold: float) -> pd.DataFrame:
    """读取行为样本，并兼容没有 label 的旧 CSV。"""
    samples = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    sort_column = "sort_key" if "sort_key" in samples.columns else "timestamp" if "timestamp" in samples.columns else None
    if sort_column is None:
        raise ValueError("Full recall eval requires sort_key or timestamp. Please rerun 01_prepare_behavior_samples.py.")
    if sort_column != "sort_key":
        samples = samples.rename(columns={sort_column: "sort_key"})

    samples["user_id"] = pd.to_numeric(samples["user_id"], errors="raise").astype("int64")
    samples["item_id"] = pd.to_numeric(samples["item_id"], errors="raise").astype("int64")
    samples["watch_ratio"] = pd.to_numeric(samples["watch_ratio"], errors="coerce").fillna(0.0).astype("float32")
    samples["sort_key"] = pd.to_numeric(samples["sort_key"], errors="coerce")
    samples = samples.dropna(subset=["sort_key"]).copy()
    samples["sort_key"] = samples["sort_key"].astype("float64")

    if "label" not in samples.columns:
        samples["label"] = np.select(
            [samples["watch_ratio"] >= pos_threshold, samples["watch_ratio"] < neg_threshold],
            [1, 0],
            default=-1,
        )
    samples["label"] = pd.to_numeric(samples["label"], errors="coerce").fillna(-1).astype("int8")
    return samples


def load_item_categories(meta_path: Path, item_ids: np.ndarray, category_to_index: dict[str, int]) -> np.ndarray:
    """按 item_ids 顺序生成 category index 数组，供全量 item tower 编码。"""
    unknown_index = int(category_to_index.get(UNKNOWN_CATEGORY, 0))
    category_indices = np.full(len(item_ids), unknown_index, dtype=np.int64)
    if not meta_path.is_file():
        return category_indices

    meta = pd.read_csv(meta_path, usecols=lambda column: column in {"item_id", "category_id"}, encoding="utf-8-sig")
    if "item_id" not in meta.columns or "category_id" not in meta.columns:
        return category_indices

    item_id_to_row = {int(item_id): row for row, item_id in enumerate(item_ids.tolist())}
    for row in meta.itertuples(index=False):
        item_id = int(row.item_id)
        if item_id not in item_id_to_row:
            continue
        category_indices[item_id_to_row[item_id]] = int(category_to_index.get(str(row.category_id), unknown_index))

    return category_indices


def build_model_from_config(config: dict, num_items: int, num_categories: int, device: torch.device):
    """用 checkpoint config 恢复模型结构。"""
    args = config["args"]
    item_tower = TextIdCategoryItemTower(
        text_dim=int(config["text_dim"]),
        num_items=num_items,
        num_categories=num_categories,
        text_proj_dim=int(args["text_proj_dim"]),
        item_id_dim=int(args["item_id_dim"]),
        category_dim=int(args["category_dim"]),
        hidden_dim=int(args["hidden_dim"]),
        output_dim=int(args["output_dim"]),
        dropout=float(args["dropout"]),
        id_feature_dropout=float(args["id_feature_dropout"]),
        normalize_output=True,
        use_item_id=not bool(args.get("no_item_id", False)),
        use_category=not bool(args.get("no_category", False)),
    ).to(device)
    user_tower = RecentHistoryUserTower(
        item_tower=item_tower,
        input_dim=int(args["output_dim"]),
        hidden_dim=int(args["hidden_dim"]),
        output_dim=int(args["output_dim"]),
        dropout=float(args["dropout"]),
        watch_weight_cap=float(args["watch_weight_cap"]),
        time_decay_days=float(args["time_decay_days"]),
        normalize_output=True,
    ).to(device)
    return item_tower, user_tower


@torch.no_grad()
def encode_all_items(
    item_tower: TextIdCategoryItemTower,
    item_text_embeddings: np.ndarray,
    item_category_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """把全量 item 编成召回向量矩阵 [N, D]。"""
    item_tower.eval()
    outputs: list[torch.Tensor] = []
    item_indices = np.arange(len(item_text_embeddings), dtype=np.int64)

    for start in tqdm(range(0, len(item_indices), batch_size), desc="encode items"):
        end = min(start + batch_size, len(item_indices))
        text = torch.from_numpy(item_text_embeddings[start:end]).to(device)
        indices = torch.from_numpy(item_indices[start:end]).to(device)
        categories = torch.from_numpy(item_category_indices[start:end]).to(device)
        item_embs = item_tower.encode_item(text, item_indices=indices, category_indices=categories)
        outputs.append(item_embs.detach().cpu())

    return torch.cat(outputs, dim=0)


def build_fixed_user_histories(
    train_samples: pd.DataFrame,
    item_id_to_index: dict[int, int],
    category_indices: np.ndarray,
    history_min_watch_ratio: float,
    max_history_len: int,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, set[int]]]:
    """用 train split 构造固定用户历史，并记录训练已正反馈 item 以便评估过滤。"""
    train_samples = train_samples[train_samples["item_id"].isin(item_id_to_index.keys())].copy()
    train_samples["item_index"] = train_samples["item_id"].map(item_id_to_index).astype("int64")
    train_samples = train_samples.sort_values(["user_id", "sort_key", "item_id"]).reset_index(drop=True)

    user_histories: dict[int, dict[str, np.ndarray]] = {}
    train_positive_items: dict[int, set[int]] = defaultdict(set)

    for user_id, group in train_samples.groupby("user_id", sort=False):
        positive_group = group[group["label"] == 1]
        for item_index in positive_group["item_index"].to_numpy(dtype=np.int64):
            train_positive_items[int(user_id)].add(int(item_index))

        history_group = group[group["watch_ratio"] >= history_min_watch_ratio].tail(max_history_len)
        if history_group.empty:
            continue

        item_indices = history_group["item_index"].to_numpy(dtype=np.int64)
        sort_keys = history_group["sort_key"].to_numpy(dtype=np.float64)
        reference_time = float(sort_keys.max())
        user_histories[int(user_id)] = {
            "item_indices": item_indices,
            "category_indices": category_indices[item_indices],
            "watch_ratios": history_group["watch_ratio"].to_numpy(dtype=np.float32),
            "time_deltas": np.maximum(reference_time - sort_keys, 0.0).astype(np.float32),
        }

    return user_histories, train_positive_items


def build_eval_relevance(eval_samples: pd.DataFrame, item_id_to_index: dict[int, int]) -> dict[int, set[int]]:
    """从 val/test 正反馈样本中构造每个用户的相关 item 集合。"""
    eval_samples = eval_samples[(eval_samples["label"] == 1) & eval_samples["item_id"].isin(item_id_to_index.keys())].copy()
    eval_samples["item_index"] = eval_samples["item_id"].map(item_id_to_index).astype("int64")

    relevance: dict[int, set[int]] = defaultdict(set)
    for user_id, item_index in zip(eval_samples["user_id"], eval_samples["item_index"], strict=False):
        relevance[int(user_id)].add(int(item_index))
    return relevance


def make_user_batch(
    user_ids: list[int],
    user_histories: dict[int, dict[str, np.ndarray]],
    item_text_embeddings: np.ndarray,
    max_history_len: int,
) -> dict[str, torch.Tensor]:
    """把一批固定用户历史 padding 成 UserTower 输入。"""
    batch_size = len(user_ids)
    text_dim = item_text_embeddings.shape[1]
    history_item_text_embs = np.zeros((batch_size, max_history_len, text_dim), dtype=np.float32)
    history_item_indices = np.zeros((batch_size, max_history_len), dtype=np.int64)
    history_category_indices = np.zeros((batch_size, max_history_len), dtype=np.int64)
    history_watch_ratios = np.zeros((batch_size, max_history_len), dtype=np.float32)
    history_time_deltas = np.zeros((batch_size, max_history_len), dtype=np.float32)
    history_mask = np.zeros((batch_size, max_history_len), dtype=np.float32)

    for row_idx, user_id in enumerate(user_ids):
        history = user_histories[user_id]
        seq_len = len(history["item_indices"])
        history_item_indices[row_idx, :seq_len] = history["item_indices"]
        history_category_indices[row_idx, :seq_len] = history["category_indices"]
        history_watch_ratios[row_idx, :seq_len] = history["watch_ratios"]
        history_time_deltas[row_idx, :seq_len] = history["time_deltas"]
        history_mask[row_idx, :seq_len] = 1.0
        history_item_text_embs[row_idx, :seq_len] = item_text_embeddings[history["item_indices"]]

    return {
        "history_item_text_embs": torch.from_numpy(history_item_text_embs),
        "history_item_indices": torch.from_numpy(history_item_indices),
        "history_category_indices": torch.from_numpy(history_category_indices),
        "history_watch_ratios": torch.from_numpy(history_watch_ratios),
        "history_time_deltas": torch.from_numpy(history_time_deltas),
        "history_mask": torch.from_numpy(history_mask),
    }


@torch.no_grad()
def rank_full_catalog(
    user_tower: RecentHistoryUserTower,
    item_embs: torch.Tensor,
    user_histories: dict[int, dict[str, np.ndarray]],
    train_positive_items: dict[int, set[int]],
    item_text_embeddings: np.ndarray,
    max_history_len: int,
    user_batch_size: int,
    top_k: int,
    device: torch.device,
) -> dict[int, list[int]]:
    """按用户批量计算全量 item topK 排名。"""
    user_tower.eval()
    item_embs = item_embs.to(device)
    user_rankings: dict[int, list[int]] = {}
    user_ids = list(user_histories.keys())

    for start in tqdm(range(0, len(user_ids), user_batch_size), desc="rank users"):
        batch_user_ids = user_ids[start:start + user_batch_size]
        batch = make_user_batch(batch_user_ids, user_histories, item_text_embeddings, max_history_len)
        batch = {key: value.to(device) for key, value in batch.items()}

        user_embs = user_tower.encode_user(**batch)
        scores = user_embs @ item_embs.T

        # 过滤训练集中已正反馈 item，避免评估把“已经看过且喜欢”的内容再次算作推荐结果。
        for row_idx, user_id in enumerate(batch_user_ids):
            seen = train_positive_items.get(int(user_id))
            if seen:
                scores[row_idx, list(seen)] = -float("inf")

        k = min(top_k, scores.shape[1])
        top_indices = torch.topk(scores, k=k, dim=1).indices.detach().cpu().numpy()
        for user_id, ranking in zip(batch_user_ids, top_indices, strict=False):
            user_rankings[int(user_id)] = [int(item_index) for item_index in ranking.tolist()]

    return user_rankings


def main() -> None:
    args = parse_args()
    ks = parse_ks(args.ks)
    max_k = max(ks)
    device = torch.device(args.device)

    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model_args = config["args"]

    embedding_config = config["embedding_config"]
    item_text_embeddings_path = resolve_project_path(embedding_config["embedding_path"])
    item_ids_path = resolve_project_path(embedding_config["item_ids_path"])
    meta_path = resolve_project_path(embedding_config.get("meta_path", ""))

    item_text_embeddings = np.load(item_text_embeddings_path).astype("float32")
    item_ids = np.load(item_ids_path).astype("int64")
    item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(item_ids.tolist())}

    category_to_index = {str(k): int(v) for k, v in config["category_to_index"].items()}
    item_category_indices = load_item_categories(meta_path, item_ids, category_to_index)

    item_tower, user_tower = build_model_from_config(config, num_items=len(item_ids), num_categories=len(category_to_index), device=device)
    item_tower.load_state_dict(checkpoint["item_tower"])
    user_tower.load_state_dict(checkpoint["user_tower"])

    pos_threshold = float(model_args["pos_threshold"])
    neg_threshold = float(model_args["neg_threshold"])
    history_min_watch_ratio = (
        float(args.history_min_watch_ratio)
        if args.history_min_watch_ratio >= 0
        else float(model_args["history_min_watch_ratio"])
    )
    max_history_len = int(model_args["max_history_len"])

    train_samples = load_behavior_samples(resolve_project_path(args.train_samples), pos_threshold, neg_threshold)
    eval_samples = load_behavior_samples(resolve_project_path(args.eval_samples), pos_threshold, neg_threshold)

    user_histories, train_positive_items = build_fixed_user_histories(
        train_samples=train_samples,
        item_id_to_index=item_id_to_index,
        category_indices=item_category_indices,
        history_min_watch_ratio=history_min_watch_ratio,
        max_history_len=max_history_len,
    )
    relevance = build_eval_relevance(eval_samples, item_id_to_index)

    # 只评估同时拥有 train history 和 eval 正反馈的用户。
    relevance = {user_id: items for user_id, items in relevance.items() if user_id in user_histories}

    print("=" * 80)
    print("V2.1 Full-Catalog Retrieval Eval")
    print("=" * 80)
    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] eval samples: {resolve_project_path(args.eval_samples)}")
    print(f"[INFO] users with history: {len(user_histories)}")
    print(f"[INFO] users with eval positives: {len(relevance)}")
    print(f"[INFO] num_items: {len(item_ids)}")
    print(f"[INFO] ks: {ks}")

    item_embs = encode_all_items(item_tower, item_text_embeddings, item_category_indices, args.batch_size, device)
    user_rankings = rank_full_catalog(
        user_tower=user_tower,
        item_embs=item_embs,
        user_histories=user_histories,
        train_positive_items=train_positive_items,
        item_text_embeddings=item_text_embeddings,
        max_history_len=max_history_len,
        user_batch_size=args.user_batch_size,
        top_k=max_k,
        device=device,
    )

    metrics = aggregate_ranking_metrics(user_rankings, relevance, ks)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
