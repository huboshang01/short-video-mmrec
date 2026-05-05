"""
V2.1 时间感知召回 Dataset。

本模块把行为样本 CSV 和 item 文本向量缓存组装成推荐召回训练样本。
和旧版 TwoTowerBehaviorDataset 最大的区别是：
    1. history 严格来自 target 时间之前，避免用未来行为预测当前 target。
    2. pointwise 样本同时支持正反馈和显式负反馈，适配 BCE 训练目标。
    3. history 按时间取最近 max_history_len 条，而不是按 watch_ratio 排全局 top50。

当前 MVP 只完整实现 TimeAwarePointwiseRetrievalDataset。Pairwise 数据集保留接口，
后续接 BPR / sampled-softmax 时再实现负样本采样策略。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


UNKNOWN_CATEGORY = "__UNK__"


@dataclass(frozen=True)
class UserHistory:
    """单个用户按时间排序后的全部可见交互。"""

    item_indices: np.ndarray
    item_ids: np.ndarray
    category_indices: np.ndarray
    watch_ratios: np.ndarray
    sort_keys: np.ndarray
    history_positions: np.ndarray


def build_category_vocab(sample_paths: Iterable[str | Path]) -> dict[str, int]:
    """
    从一组行为样本文件中构建统一 category vocab。

    训练集、验证集和测试集必须共享同一套 category_id -> index 映射；否则同一个类目在
    不同 Dataset 中会对应不同 embedding 行，评估和 checkpoint 加载都会错位。
    """
    categories: set[str] = {UNKNOWN_CATEGORY}
    for path in sample_paths:
        path = Path(path)
        if not path.exists():
            continue

        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        if "category_id" not in header.columns:
            continue

        values = pd.read_csv(path, usecols=["category_id"], encoding="utf-8-sig")
        categories.update(values["category_id"].fillna(UNKNOWN_CATEGORY).astype(str).tolist())

    return {category: idx for idx, category in enumerate(sorted(categories))}


class TimeAwarePointwiseRetrievalDataset(Dataset):
    """
    时间感知 pointwise 召回数据集。

    单条样本结构：
        history: target 发生之前的最近若干条用户兴趣历史。
        target: 当前要判断是否推荐的 item。
        label:  1 表示正反馈，0 表示显式负反馈。

    训练目标：
        user tower(history) 与 item tower(target) 的 dot score 进入 BCEWithLogitsLoss。
    """

    def __init__(
        self,
        behavior_samples_path: str | Path,
        item_embeddings_path: str | Path,
        item_ids_path: str | Path,
        category_to_index: dict[str, int] | None = None,
        max_history_len: int = 50,
        pos_threshold: float = 1.0,
        neg_threshold: float = 0.7,
        history_min_watch_ratio: float = 1.0,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if max_history_len <= 0:
            raise ValueError("max_history_len must be positive.")
        if neg_threshold >= pos_threshold:
            raise ValueError("neg_threshold must be smaller than pos_threshold.")
        if history_min_watch_ratio < 0:
            raise ValueError("history_min_watch_ratio must be non-negative.")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")

        self.behavior_samples_path = Path(behavior_samples_path)
        self.item_embeddings_path = Path(item_embeddings_path)
        self.item_ids_path = Path(item_ids_path)
        self.max_history_len = int(max_history_len)
        self.pos_threshold = float(pos_threshold)
        self.neg_threshold = float(neg_threshold)
        self.history_min_watch_ratio = float(history_min_watch_ratio)

        self.item_text_embeddings, self.item_ids = self._load_item_embedding_cache()
        self.item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(self.item_ids.tolist())}

        self.category_to_index = self._normalize_category_vocab(category_to_index)

        behavior_samples = self._load_behavior_samples()
        behavior_samples = self._attach_item_and_category_indices(behavior_samples)

        self.user_histories, target_samples = self._build_histories_and_targets(behavior_samples)
        if max_samples is not None and len(target_samples) > max_samples:
            target_samples = target_samples.sample(n=max_samples, random_state=seed, replace=False).reset_index(drop=True)

        if target_samples.empty:
            raise ValueError(
                "No valid pointwise target samples. "
                "Please check timestamp/sort_key, label thresholds and history_min_watch_ratio."
            )

        self.target_user_ids = target_samples["user_id"].to_numpy(dtype=np.int64)
        self.target_item_ids = target_samples["item_id"].to_numpy(dtype=np.int64)
        self.target_item_indices = target_samples["item_index"].to_numpy(dtype=np.int64)
        self.target_category_indices = target_samples["category_index"].to_numpy(dtype=np.int64)
        self.target_positions = target_samples["target_position"].to_numpy(dtype=np.int64)
        self.target_sort_keys = target_samples["sort_key"].to_numpy(dtype=np.float64)
        self.labels = target_samples["label"].to_numpy(dtype=np.float32)
        self.target_watch_ratios = target_samples["watch_ratio"].to_numpy(dtype=np.float32)
        self.sample_weights = target_samples["sample_weight"].to_numpy(dtype=np.float32)

    def _load_item_embedding_cache(self) -> tuple[np.ndarray, np.ndarray]:
        """读取 BGE item 文本向量缓存，并校验 item_ids 与向量行一一对应。"""
        item_text_embeddings = np.load(self.item_embeddings_path).astype("float32")
        item_ids = np.load(self.item_ids_path).astype("int64")

        if item_text_embeddings.ndim != 2:
            raise ValueError(f"item_text_embeddings must be 2D, got {item_text_embeddings.shape}.")
        if item_ids.ndim != 1:
            raise ValueError(f"item_ids must be 1D, got {item_ids.shape}.")
        if item_text_embeddings.shape[0] != item_ids.shape[0]:
            raise ValueError("item_text_embeddings and item_ids row count mismatch.")

        return np.ascontiguousarray(item_text_embeddings), item_ids

    def _normalize_category_vocab(self, category_to_index: dict[str, int] | None) -> dict[str, int]:
        """确保 category vocab 至少包含 UNKNOWN_CATEGORY。"""
        if category_to_index is None:
            return {UNKNOWN_CATEGORY: 0}

        normalized = {str(category): int(index) for category, index in category_to_index.items()}
        if UNKNOWN_CATEGORY not in normalized:
            normalized[UNKNOWN_CATEGORY] = 0
        return normalized

    def _load_behavior_samples(self) -> pd.DataFrame:
        """读取行为样本并补齐 label / sample_weight 等训练字段。"""
        if not self.behavior_samples_path.exists():
            raise FileNotFoundError(f"behavior_samples_path not found: {self.behavior_samples_path}")

        behavior_samples = pd.read_csv(self.behavior_samples_path, encoding="utf-8-sig", low_memory=False)

        required = {"user_id", "item_id", "watch_ratio", "category_id"}
        missing = required - set(behavior_samples.columns)
        if missing:
            raise ValueError(f"Missing required behavior columns: {missing}")

        sort_column = self._resolve_sort_column(behavior_samples)
        behavior_samples = behavior_samples.rename(columns={sort_column: "sort_key"}).copy()

        behavior_samples["user_id"] = pd.to_numeric(behavior_samples["user_id"], errors="raise").astype("int64")
        behavior_samples["item_id"] = pd.to_numeric(behavior_samples["item_id"], errors="raise").astype("int64")
        behavior_samples["watch_ratio"] = (
            pd.to_numeric(behavior_samples["watch_ratio"], errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )
        behavior_samples["sort_key"] = pd.to_numeric(behavior_samples["sort_key"], errors="coerce")
        behavior_samples = behavior_samples.dropna(subset=["sort_key"]).copy()
        behavior_samples["sort_key"] = behavior_samples["sort_key"].astype("float64")
        behavior_samples["category_id"] = behavior_samples["category_id"].fillna(UNKNOWN_CATEGORY).astype(str)

        if "label" not in behavior_samples.columns:
            # 兼容重新生成前的样本文件：没有 label 时按阈值即时推导。
            behavior_samples["label"] = np.select(
                [
                    behavior_samples["watch_ratio"] >= self.pos_threshold,
                    behavior_samples["watch_ratio"] < self.neg_threshold,
                ],
                [1, 0],
                default=-1,
            )
        behavior_samples["label"] = pd.to_numeric(behavior_samples["label"], errors="coerce").fillna(-1).astype("int8")

        if "sample_weight" not in behavior_samples.columns:
            clipped_watch = behavior_samples["watch_ratio"].clip(lower=0.0, upper=5.0)
            behavior_samples["sample_weight"] = np.where(
                behavior_samples["label"] == 1,
                np.log1p(clipped_watch),
                1.0,
            )
        behavior_samples["sample_weight"] = (
            pd.to_numeric(behavior_samples["sample_weight"], errors="coerce")
            .fillna(1.0)
            .astype("float32")
        )

        return behavior_samples

    def _resolve_sort_column(self, behavior_samples: pd.DataFrame) -> str:
        """优先使用 sort_key，其次 timestamp；没有时间字段时直接报错，避免静默时间泄漏。"""
        for candidate in ("sort_key", "timestamp"):
            if candidate in behavior_samples.columns:
                return candidate
        raise ValueError(
            "TimeAwarePointwiseRetrievalDataset requires sort_key or timestamp. "
            "Please rerun scripts/v2/01_prepare_behavior_samples.py."
        )

    def _attach_item_and_category_indices(self, behavior_samples: pd.DataFrame) -> pd.DataFrame:
        """把原始 item_id/category_id 映射成 embedding table 可直接索引的整数。"""
        behavior_samples = behavior_samples[behavior_samples["item_id"].isin(self.item_id_to_index.keys())].copy()
        behavior_samples["item_index"] = behavior_samples["item_id"].map(self.item_id_to_index).astype("int64")

        unknown_index = self.category_to_index[UNKNOWN_CATEGORY]
        behavior_samples["category_index"] = (
            behavior_samples["category_id"]
            .map(lambda category: self.category_to_index.get(str(category), unknown_index))
            .astype("int64")
        )

        return behavior_samples

    def _build_histories_and_targets(
        self,
        behavior_samples: pd.DataFrame,
    ) -> tuple[dict[int, UserHistory], pd.DataFrame]:
        """
        为每个用户建立时间序列，同时收集有历史的 pointwise target。

        target 的 history 只允许使用 target_position 之前、且 watch_ratio 达到
        history_min_watch_ratio 的交互。这样负样本 target 不会被混入用户兴趣表示。
        """
        user_histories: dict[int, UserHistory] = {}
        target_frames: list[pd.DataFrame] = []

        sorted_samples = behavior_samples.sort_values(["user_id", "sort_key", "item_id"]).reset_index(drop=True)

        for user_id, group in sorted_samples.groupby("user_id", sort=False):
            group = group.reset_index(drop=True)

            item_indices = group["item_index"].to_numpy(dtype=np.int64)
            item_ids = group["item_id"].to_numpy(dtype=np.int64)
            category_indices = group["category_index"].to_numpy(dtype=np.int64)
            watch_ratios = group["watch_ratio"].to_numpy(dtype=np.float32)
            sort_keys = group["sort_key"].to_numpy(dtype=np.float64)

            history_mask = watch_ratios >= self.history_min_watch_ratio
            history_positions = np.flatnonzero(history_mask).astype(np.int64)

            user_histories[int(user_id)] = UserHistory(
                item_indices=item_indices,
                item_ids=item_ids,
                category_indices=category_indices,
                watch_ratios=watch_ratios,
                sort_keys=sort_keys,
                history_positions=history_positions,
            )

            target_positions = np.flatnonzero(group["label"].isin([0, 1]).to_numpy()).astype(np.int64)
            if len(target_positions) == 0:
                continue

            # 向量化判断 target 之前是否已经有可用兴趣历史，避免逐行 append 拖慢全量构造。
            history_ends = np.searchsorted(history_positions, target_positions, side="left")
            valid_target_positions = target_positions[history_ends > 0]
            if len(valid_target_positions) == 0:
                continue

            target_frame = group.iloc[valid_target_positions][
                [
                    "user_id",
                    "item_id",
                    "item_index",
                    "category_index",
                    "sort_key",
                    "watch_ratio",
                    "label",
                    "sample_weight",
                ]
            ].copy()
            target_frame["target_position"] = valid_target_positions
            target_frames.append(target_frame)

        if not target_frames:
            return user_histories, pd.DataFrame()

        return user_histories, pd.concat(target_frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_id = int(self.target_user_ids[idx])
        target_item_index = int(self.target_item_indices[idx])
        target_category_index = int(self.target_category_indices[idx])
        target_sort_key = float(self.target_sort_keys[idx])
        target_position = int(self.target_positions[idx])

        history = self.user_histories[user_id]

        # 从该用户可作为兴趣历史的位置中，截取 target 之前最近 max_history_len 条。
        history_end = np.searchsorted(history.history_positions, target_position, side="left")
        history_start = max(0, history_end - self.max_history_len)
        selected_positions = history.history_positions[history_start:history_end]

        seq_len = len(selected_positions)
        if seq_len == 0:
            raise RuntimeError(f"User {user_id} has no causal history for target index {target_item_index}.")

        padded_item_indices = np.zeros(self.max_history_len, dtype=np.int64)
        padded_item_ids = np.zeros(self.max_history_len, dtype=np.int64)
        padded_category_indices = np.zeros(self.max_history_len, dtype=np.int64)
        padded_watch_ratios = np.zeros(self.max_history_len, dtype=np.float32)
        padded_time_deltas = np.zeros(self.max_history_len, dtype=np.float32)
        history_mask = np.zeros(self.max_history_len, dtype=np.float32)

        padded_item_indices[:seq_len] = history.item_indices[selected_positions]
        padded_item_ids[:seq_len] = history.item_ids[selected_positions]
        padded_category_indices[:seq_len] = history.category_indices[selected_positions]
        padded_watch_ratios[:seq_len] = history.watch_ratios[selected_positions]

        # delta 越小代表越近的行为；UserTower 会基于它做时间衰减。
        time_deltas = np.maximum(target_sort_key - history.sort_keys[selected_positions], 0.0)
        padded_time_deltas[:seq_len] = time_deltas.astype(np.float32)
        history_mask[:seq_len] = 1.0

        history_item_text_embs = self.item_text_embeddings[padded_item_indices]
        target_item_text_emb = self.item_text_embeddings[target_item_index]

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "history_item_ids": torch.from_numpy(padded_item_ids),
            "history_item_indices": torch.from_numpy(padded_item_indices),
            "history_category_indices": torch.from_numpy(padded_category_indices),
            "history_item_text_embs": torch.from_numpy(history_item_text_embs),
            "history_watch_ratios": torch.from_numpy(padded_watch_ratios),
            "history_time_deltas": torch.from_numpy(padded_time_deltas),
            "history_mask": torch.from_numpy(history_mask),
            "target_item_id": torch.tensor(int(self.target_item_ids[idx]), dtype=torch.long),
            "target_item_index": torch.tensor(target_item_index, dtype=torch.long),
            "target_category_index": torch.tensor(target_category_index, dtype=torch.long),
            "target_item_text_emb": torch.from_numpy(target_item_text_emb),
            "target_watch_ratio": torch.tensor(float(self.target_watch_ratios[idx]), dtype=torch.float32),
            "label": torch.tensor(float(self.labels[idx]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32),
        }


class TimeAwarePairwiseRetrievalDataset(Dataset):
    """
    时间感知 pairwise 召回数据集接口。

    后续用于：
        - BPR: history + positive target + sampled negative target。
        - sampled-softmax: history + positive target + 多个负样本 target。

    该版本暂不实现，避免 MVP 阶段把 BCE 与 pairwise 采样逻辑混在一起。
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "TimeAwarePairwiseRetrievalDataset is reserved for V2.2 BPR/sampled-softmax."
        )

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError(idx)
