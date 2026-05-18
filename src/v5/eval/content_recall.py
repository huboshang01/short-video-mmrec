"""基于 item 内容向量的 full-catalog 用户召回评估。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv

import numpy as np

from src.v3.eval.retrieval_metrics import aggregate_ranking_metrics
from src.v5.retrieval.embeddings import l2_normalize


def read_positive_samples(path: Path, item_id_to_row: dict[int, int]) -> list[dict[str, int]]:
    """读取 label=1 的正反馈，并把真实 item_id 映射成 item_vectors 的行号。"""
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(float(row["label"])) != 1:
                continue
            item_id = int(row["item_id"])
            if item_id in item_id_to_row:
                rows.append(
                    {
                        "user_id": int(row["user_id"]),
                        "item_row": item_id_to_row[item_id],
                        "sort_key": int(row["sort_key"]),
                    }
                )
    return rows


def build_histories(
    train_rows: list[dict[str, int]],
    max_history_len: int,
) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
    """按用户构造训练历史和已看集合。

    histories 用最近 max_history_len 个正反馈 item 生成用户向量；
    seen 用于评估时过滤 train 中已经正反馈过的 item。
    """
    grouped: dict[int, list[dict[str, int]]] = defaultdict(list)
    seen: dict[int, set[int]] = defaultdict(set)
    for row in train_rows:
        grouped[row["user_id"]].append(row)
        seen[row["user_id"]].add(row["item_row"])
    histories = {}
    for user_id, rows in grouped.items():
        rows.sort(key=lambda x: (x["sort_key"], x["item_row"]))
        histories[user_id] = [row["item_row"] for row in rows[-max_history_len:]]
    return histories, seen


def build_relevance(eval_rows: list[dict[str, int]]) -> dict[int, set[int]]:
    """把 val/test 正反馈整理成每个用户的相关 item 集合。"""
    relevance: dict[int, set[int]] = defaultdict(set)
    for row in eval_rows:
        relevance[row["user_id"]].add(row["item_row"])
    return relevance


def rank_by_mean_history(
    item_vectors: np.ndarray,
    histories: dict[int, list[int]],
    seen: dict[int, set[int]],
    top_k: int,
    user_batch_size: int,
) -> dict[int, list[int]]:
    """用历史均值用户向量对全量 item 做精确 ranking。"""
    item_vectors = l2_normalize(item_vectors)
    rankings = {}
    user_ids = list(histories.keys())
    k = min(top_k, item_vectors.shape[0])

    for start in range(0, len(user_ids), user_batch_size):
        batch_user_ids = user_ids[start : start + user_batch_size]
        user_vectors = []
        for user_id in batch_user_ids:
            # 最小内容召回画像：用户向量取历史 item 内容向量均值。
            user_vectors.append(item_vectors[histories[user_id]].mean(axis=0))
        user_vectors = l2_normalize(np.asarray(user_vectors, dtype="float32"))

        # item_vectors 已归一化，因此点积就是 cosine similarity。
        scores = user_vectors @ item_vectors.T

        # 过滤 train 已交互 item，和 V3/V4 full-catalog 评估口径保持一致。
        for row_idx, user_id in enumerate(batch_user_ids):
            if seen.get(user_id):
                scores[row_idx, list(seen[user_id])] = -np.inf

        # argpartition 先取候选 topK，再在候选内部精排，避免全量排序。
        candidates = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        top_indices = np.take_along_axis(candidates, order, axis=1)
        for user_id, ranking in zip(batch_user_ids, top_indices, strict=False):
            rankings[int(user_id)] = [int(idx) for idx in ranking.tolist()]
    return rankings


def evaluate_content_recall(
    item_vectors: np.ndarray,
    item_ids: list[int],
    train_path: Path,
    eval_path: Path,
    ks: list[int],
    max_history_len: int,
    max_eval_users: int | None = None,
    user_batch_size: int = 256,
) -> dict[str, float]:
    """端到端评估一路内容向量召回。

    输入 item_vectors 和其并行 item_ids，输出和 V3 一致的 Recall/NDCG/MRR/HitRate。
    """
    # real item_id -> item_vectors row，行为样本中的 item_id 先统一转为矩阵行号。
    item_id_to_row = {int(item_id): idx for idx, item_id in enumerate(item_ids)}
    train_rows = read_positive_samples(train_path, item_id_to_row)
    eval_rows = read_positive_samples(eval_path, item_id_to_row)
    histories, seen = build_histories(train_rows, max_history_len)
    relevance = build_relevance(eval_rows)

    # 只评估同时拥有 train 历史和 eval 正反馈的用户。
    relevance = {user_id: items for user_id, items in relevance.items() if user_id in histories}
    if max_eval_users is not None:
        keep = sorted(relevance)[:max_eval_users]
        relevance = {user_id: relevance[user_id] for user_id in keep}
        histories = {user_id: histories[user_id] for user_id in keep}
        seen = {user_id: seen[user_id] for user_id in keep}
    rankings = rank_by_mean_history(
        item_vectors=item_vectors,
        histories=histories,
        seen=seen,
        top_k=max(ks),
        user_batch_size=user_batch_size,
    )
    metrics = aggregate_ranking_metrics(rankings, relevance, ks)
    metrics["candidate_items"] = float(len(item_ids))
    return metrics
