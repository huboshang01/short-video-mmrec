"""
V3 Step 03: MicroLens-100K 全量 item 召回评估。

评估口径：
    1. 用 train split 构造固定用户历史。
    2. 编码全量 item，和每个 user embedding 做矩阵乘法。
    3. 过滤 train 已交互 item。
    4. 用 val/test 的正反馈 item 计算 Recall/NDCG/MRR/HitRate。

这和线上召回更接近：用户画像固定后，从全量 item 库中取 topK。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.data.microlens_retrieval_dataset import MultimodalFeatureCache, load_item_ids
from src.v3.eval.retrieval_metrics import aggregate_ranking_metrics
from src.v3.models.multimodal_item_encoder import MultimodalItemEncoder
from src.v3.models.user_tower import RecentHistoryUserTower


DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "v3" / "microlens_100k" / "retrieval_mvp" / "retrieval_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V3 MicroLens retrieval checkpoint with full-catalog ranking.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to V3 checkpoint.")
    parser.add_argument("--eval-split", type=str, default="test", choices=["val", "test"], help="Which split to evaluate.")
    parser.add_argument("--ks", type=str, default="", help="Comma-separated K values. Empty means config eval.ks.")
    parser.add_argument("--output", type=str, default="", help="Optional metrics JSON output path.")
    parser.add_argument("--batch-size", type=int, default=-1, help="Override item encoding batch size when > 0.")
    parser.add_argument("--user-batch-size", type=int, default=-1, help="Override user ranking batch size when > 0.")
    parser.add_argument("--max-eval-users", type=int, default=None, help="Limit evaluated users for quick smoke tests; -1 means all.")
    parser.add_argument("--device", type=str, default="", help="Override device, e.g. cuda or cpu.")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_ks(value: str, fallback: list[int]) -> list[int]:
    if not value:
        return sorted(set(int(k) for k in fallback))
    ks = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ks or min(ks) <= 0:
        raise ValueError("--ks must contain positive integers.")
    return sorted(set(ks))


def resolve_device(device_arg: str) -> torch.device:
    """支持 config 中的 auto，并在 CUDA 不可用时回退到 CPU。"""
    device_name = device_arg
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def build_model_from_config(train_config: dict, device: torch.device):
    config = train_config["config"]
    model_cfg = config["model"]
    dims = train_config["feature_dims"]
    item_tower = MultimodalItemEncoder(
        text_dim=int(dims["text"]),
        image_dim=int(dims["image"]),
        video_dim=int(dims["video"]),
        num_items=int(train_config["num_items"]),
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


@torch.no_grad()
def encode_all_items(
    item_tower: MultimodalItemEncoder,
    feature_cache: MultimodalFeatureCache,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """把全量 item 编码成 [N, D]，后续用户打分直接矩阵乘法。"""
    item_tower.eval()
    outputs: list[torch.Tensor] = []
    for start in tqdm(range(0, feature_cache.num_items, batch_size), desc="encode items"):
        end = min(start + batch_size, feature_cache.num_items)
        indices = torch.arange(start, end, dtype=torch.long, device=device)
        features = feature_cache.gather(indices, device=device)
        item_embs = item_tower.encode_item(**features, item_indices=indices)
        outputs.append(item_embs.detach().cpu())
    return torch.cat(outputs, dim=0)


def read_positive_samples(path: Path, item_id_to_index: dict[int, int]) -> list[dict[str, int]]:
    """读取 label=1 的行为样本，并转换 item_id 为 item_index。"""
    rows: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row["label"])) != 1:
                continue
            item_id = int(row["item_id"])
            if item_id not in item_id_to_index:
                continue
            rows.append(
                {
                    "user_id": int(row["user_id"]),
                    "item_index": int(item_id_to_index[item_id]),
                    "sort_key": int(row["sort_key"]),
                }
            )
    return rows


def build_fixed_user_histories(
    train_rows: list[dict[str, int]],
    max_history_len: int,
) -> tuple[dict[int, np.ndarray], dict[int, set[int]]]:
    """用 train split 构造固定用户历史，并记录需要过滤的已交互 item。"""
    grouped: dict[int, list[dict[str, int]]] = defaultdict(list)
    train_seen_items: dict[int, set[int]] = defaultdict(set)
    for row in train_rows:
        user_id = int(row["user_id"])
        grouped[user_id].append(row)
        train_seen_items[user_id].add(int(row["item_index"]))

    user_histories: dict[int, np.ndarray] = {}
    for user_id, rows in grouped.items():
        rows.sort(key=lambda row: (int(row["sort_key"]), int(row["item_index"])))
        item_indices = [int(row["item_index"]) for row in rows[-max_history_len:]]
        if item_indices:
            user_histories[user_id] = np.asarray(item_indices, dtype=np.int64)
    return user_histories, train_seen_items


def build_eval_relevance(eval_rows: list[dict[str, int]]) -> dict[int, set[int]]:
    """将 val/test 正反馈聚合成每个用户的相关 item 集合。"""
    relevance: dict[int, set[int]] = defaultdict(set)
    for row in eval_rows:
        relevance[int(row["user_id"])].add(int(row["item_index"]))
    return relevance


def make_user_batch(user_ids: list[int], user_histories: dict[int, np.ndarray], max_history_len: int) -> dict[str, torch.Tensor]:
    """把一批用户历史 padding 成 UserTower 输入。"""
    history_item_indices = np.zeros((len(user_ids), max_history_len), dtype=np.int64)
    history_mask = np.zeros((len(user_ids), max_history_len), dtype=np.float32)

    for row_idx, user_id in enumerate(user_ids):
        history = user_histories[user_id]
        seq_len = len(history)
        history_item_indices[row_idx, :seq_len] = history
        history_mask[row_idx, :seq_len] = 1.0

    return {
        "history_item_indices": torch.from_numpy(history_item_indices),
        "history_mask": torch.from_numpy(history_mask),
    }


@torch.no_grad()
def rank_full_catalog(
    item_tower: MultimodalItemEncoder,
    user_tower: RecentHistoryUserTower,
    feature_cache: MultimodalFeatureCache,
    item_embs: torch.Tensor,
    user_histories: dict[int, np.ndarray],
    train_seen_items: dict[int, set[int]],
    user_batch_size: int,
    max_history_len: int,
    top_k: int,
    device: torch.device,
) -> dict[int, list[int]]:
    """批量计算用户对全量 item 的 topK 排名。"""
    item_tower.eval()
    user_tower.eval()
    item_embs = item_embs.to(device)
    rankings: dict[int, list[int]] = {}
    user_ids = list(user_histories.keys())

    for start in tqdm(range(0, len(user_ids), user_batch_size), desc="rank users"):
        batch_user_ids = user_ids[start:start + user_batch_size]
        batch = make_user_batch(batch_user_ids, user_histories, max_history_len)
        history_indices = batch["history_item_indices"].to(device)
        history_mask = batch["history_mask"].to(device)

        history_features = feature_cache.gather(history_indices, device=device)
        history_item_embs = item_tower.encode_item(**history_features, item_indices=history_indices)
        user_embs = user_tower.encode_user(history_item_embs=history_item_embs, history_mask=history_mask)
        scores = user_embs @ item_embs.T

        # 过滤 train 已交互 item，避免把用户已经看过的内容算作推荐命中。
        for row_idx, user_id in enumerate(batch_user_ids):
            seen = train_seen_items.get(int(user_id))
            if seen:
                scores[row_idx, list(seen)] = -float("inf")

        k = min(top_k, scores.shape[1])
        top_indices = torch.topk(scores, k=k, dim=1).indices.detach().cpu().numpy()
        for user_id, ranking in zip(batch_user_ids, top_indices, strict=False):
            rankings[int(user_id)] = [int(item_index) for item_index in ranking.tolist()]

    return rankings


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = checkpoint["config"]
    config = train_config["config"]

    device = resolve_device(args.device or str(config["train"].get("device", "auto")))

    ks = parse_ks(args.ks, fallback=list(config["eval"]["ks"]))
    batch_size = args.batch_size if args.batch_size > 0 else int(config["eval"]["batch_size"])
    user_batch_size = args.user_batch_size if args.user_batch_size > 0 else int(config["eval"]["user_batch_size"])
    max_eval_users_value = args.max_eval_users if args.max_eval_users is not None else int(config["eval"].get("max_eval_users", -1))
    max_eval_users = None if int(max_eval_users_value) == -1 else int(max_eval_users_value)

    feature_cache = MultimodalFeatureCache.from_config(resolve_project_path(config["data"]["feature_config"]))
    feature_cache.validate()
    item_ids = load_item_ids(resolve_project_path(config["data"]["item_ids"]))
    item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(item_ids)}

    item_tower, user_tower = build_model_from_config(train_config, device)
    item_tower.load_state_dict(checkpoint["item_tower"])
    user_tower.load_state_dict(checkpoint["user_tower"])

    train_rows = read_positive_samples(resolve_project_path(config["data"]["train_samples"]), item_id_to_index)
    eval_path = resolve_project_path(config["data"][f"{args.eval_split}_samples"])
    eval_rows = read_positive_samples(eval_path, item_id_to_index)

    max_history_len = int(config["train"]["max_history_len"])
    user_histories, train_seen_items = build_fixed_user_histories(train_rows, max_history_len=max_history_len)
    relevance = build_eval_relevance(eval_rows)
    relevance = {user_id: items for user_id, items in relevance.items() if user_id in user_histories}
    if max_eval_users is not None:
        keep_users = sorted(relevance.keys())[:max_eval_users]
        relevance = {user_id: relevance[user_id] for user_id in keep_users}
        user_histories = {user_id: user_histories[user_id] for user_id in keep_users}

    print("=" * 80)
    print("V3 Full-Catalog Retrieval Eval")
    print("=" * 80)
    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] eval split: {args.eval_split}")
    print(f"[INFO] eval samples: {eval_path}")
    print(f"[INFO] users with train history: {len(user_histories)}")
    print(f"[INFO] users with eval positives: {len(relevance)}")
    print(f"[INFO] num_items: {feature_cache.num_items}")
    print(f"[INFO] ks: {ks}")

    item_embs = encode_all_items(item_tower, feature_cache, batch_size=batch_size, device=device)
    rankings = rank_full_catalog(
        item_tower=item_tower,
        user_tower=user_tower,
        feature_cache=feature_cache,
        item_embs=item_embs,
        user_histories=user_histories,
        train_seen_items=train_seen_items,
        user_batch_size=user_batch_size,
        max_history_len=max_history_len,
        top_k=max(ks),
        device=device,
    )

    metrics = aggregate_ranking_metrics(rankings, relevance, ks)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    output_path = resolve_project_path(args.output) if args.output else checkpoint_path.parent / f"full_recall_{args.eval_split}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
