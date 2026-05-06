"""
V2.1 全量召回评估指标。

召回模型最终要解决的是“从全量 item 库里给用户找回可能喜欢的 item”，因此评估必须使用
full-catalog ranking，而不是旧版 in-batch Acc。这里提供常用 Recall@K、HitRate@K、
NDCG@K 和 MRR@K 的集合版本实现。
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _as_relevance_set(relevant_items: Sequence[int] | set[int]) -> set[int]:
    """把相关 item 序列转成 set，方便 O(1) 命中判断。"""
    return relevant_items if isinstance(relevant_items, set) else set(int(item) for item in relevant_items)


def recall_at_k(ranked_items: Sequence[int], relevant_items: Sequence[int] | set[int], k: int) -> float:
    """Recall@K = topK 命中的相关 item 数 / 该用户所有相关 item 数。"""
    if k <= 0:
        raise ValueError("k must be positive.")
    relevant = _as_relevance_set(relevant_items)
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked_items[:k] if int(item) in relevant)
    return hits / len(relevant)


def hit_rate_at_k(ranked_items: Sequence[int], relevant_items: Sequence[int] | set[int], k: int) -> float:
    """HitRate@K 只关心 topK 里是否至少命中一个相关 item。"""
    if k <= 0:
        raise ValueError("k must be positive.")
    relevant = _as_relevance_set(relevant_items)
    return float(any(int(item) in relevant for item in ranked_items[:k]))


def ndcg_at_k(ranked_items: Sequence[int], relevant_items: Sequence[int] | set[int], k: int) -> float:
    """NDCG@K 会给排名靠前的命中更高权重。"""
    if k <= 0:
        raise ValueError("k must be positive.")
    relevant = _as_relevance_set(relevant_items)
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if int(item) in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_items: Sequence[int], relevant_items: Sequence[int] | set[int], k: int) -> float:
    """MRR@K 只看第一个相关 item 的倒数排名。"""
    if k <= 0:
        raise ValueError("k must be positive.")
    relevant = _as_relevance_set(relevant_items)
    for rank, item in enumerate(ranked_items[:k], start=1):
        if int(item) in relevant:
            return 1.0 / rank
    return 0.0


def aggregate_ranking_metrics(
    user_rankings: dict[int, Sequence[int]],
    user_relevant_items: dict[int, Sequence[int] | set[int]],
    ks: Sequence[int],
) -> dict[str, float]:
    """对多个用户的 ranking 结果求平均指标。"""
    if not ks:
        raise ValueError("ks must not be empty.")

    metric_sums: dict[str, float] = {}
    evaluated_users = 0

    for user_id, relevant_items in user_relevant_items.items():
        relevant = _as_relevance_set(relevant_items)
        ranked_items = user_rankings.get(int(user_id))
        if not relevant or ranked_items is None:
            continue

        evaluated_users += 1
        for k in ks:
            metric_sums[f"recall@{k}"] = metric_sums.get(f"recall@{k}", 0.0) + recall_at_k(ranked_items, relevant, k)
            metric_sums[f"hitrate@{k}"] = metric_sums.get(f"hitrate@{k}", 0.0) + hit_rate_at_k(ranked_items, relevant, k)
            metric_sums[f"ndcg@{k}"] = metric_sums.get(f"ndcg@{k}", 0.0) + ndcg_at_k(ranked_items, relevant, k)
            metric_sums[f"mrr@{k}"] = metric_sums.get(f"mrr@{k}", 0.0) + mrr_at_k(ranked_items, relevant, k)

    denom = max(evaluated_users, 1)
    metrics = {name: value / denom for name, value in metric_sums.items()}
    metrics["evaluated_users"] = float(evaluated_users)
    return metrics
