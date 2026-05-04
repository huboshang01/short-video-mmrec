"""
V2 双塔 InfoNCE 数据集构造。

本模块负责把行为样本 CSV 和 item 文本向量缓存拼成训练 batch：
用户侧输入是历史 item 的文本向量序列，item 侧输入是当前正样本 item 的文本向量。
负样本不在 dataset 中显式采样，而是在 loss 中直接使用同 batch 的其他目标 item。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"user_id", "item_id", "watch_ratio", "is_positive"}


class TwoTowerBehaviorDataset(Dataset):
    """
    V2 双塔召回训练数据集。

    单条样本包含：
        - history_item_text_embs: [L, D]，用户历史 item 的 BGE 文本向量。
        - watch_ratios: [L]，历史 item 对应观看比例，用作 UserEncoder 池化权重。
        - mask: [L]，历史序列 padding mask。
        - target_item_text_emb: [D]，当前正样本 item 的 BGE 文本向量。

    目标样本：
        默认只保留 is_positive == 1 的高观看比例 item 作为正样本。

    负样本：
        不额外构造，InfoNCE 会把同 batch 其他 target item 当作 in-batch negatives。
    """

    def __init__(
        self,
        behavior_samples_path: str,
        item_embeddings_path: str,
        item_ids_path: str,
        max_history_len: int = 50,
        only_positive_targets: bool = True,
        history_min_watch_ratio: float = 0.0,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        super().__init__()

        if max_history_len <= 0:
            raise ValueError("max_history_len must be positive.")
        if history_min_watch_ratio < 0:
            raise ValueError("history_min_watch_ratio must be non-negative.")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")

        self.behavior_samples_path = Path(behavior_samples_path)
        self.item_embeddings_path = Path(item_embeddings_path)
        self.item_ids_path = Path(item_ids_path)
        self.max_history_len = max_history_len
        self.only_positive_targets = only_positive_targets
        self.history_min_watch_ratio = history_min_watch_ratio

        self.item_text_embeddings, self.item_ids = self._load_item_embedding_cache()
        self.item_id_to_index = {
            int(item_id): idx for idx, item_id in enumerate(self.item_ids.tolist())
        }

        behavior_samples = self._load_behavior_samples()
        behavior_samples = behavior_samples[
            behavior_samples["item_id"].isin(self.item_id_to_index.keys())
        ].copy()
        behavior_samples["item_index"] = (
            behavior_samples["item_id"].map(self.item_id_to_index).astype("int64")
        )

        self.user_histories = self._build_user_histories(behavior_samples)
        
        target_samples = self._build_target_samples(
            behavior_samples=behavior_samples,
            max_samples=max_samples,
            seed=seed,
        )

        self.target_user_ids = target_samples["user_id"].to_numpy(dtype=np.int64)
        self.target_item_ids = target_samples["item_id"].to_numpy(dtype=np.int64)
        self.target_item_indices = target_samples["item_index"].to_numpy(dtype=np.int64)
        self.target_watch_ratios = target_samples["watch_ratio"].to_numpy(dtype=np.float32)

        if len(self.target_user_ids) == 0:
            raise ValueError("No valid target samples found for two-tower training.")

    def _load_item_embedding_cache(self) -> tuple[np.ndarray, np.ndarray]:
        """读取 item 文本向量缓存，并校验 item_ids 与向量行数一一对应。"""
        item_text_embeddings = np.load(self.item_embeddings_path).astype("float32")
        item_ids = np.load(self.item_ids_path).astype("int64")

        if item_text_embeddings.ndim != 2:
            raise ValueError(
                f"item_text_embeddings must be 2D, got {item_text_embeddings.shape}."
            )
        if item_ids.ndim != 1:
            raise ValueError(f"item_ids must be 1D, got {item_ids.shape}.")
        if item_text_embeddings.shape[0] != item_ids.shape[0]:
            raise ValueError("item_text_embeddings and item_ids row count mismatch.")

        # ascontiguousarray 保证后续 torch.from_numpy 不会遇到非连续内存带来的额外拷贝。
        return np.ascontiguousarray(item_text_embeddings), item_ids

    def _load_behavior_samples(self) -> pd.DataFrame:
        """读取行为样本，只保留双塔训练真正需要的列。"""
        behavior_samples = pd.read_csv(
            self.behavior_samples_path,
            usecols=lambda column: column in REQUIRED_COLUMNS,
            encoding="utf-8-sig",
        )

        missing = REQUIRED_COLUMNS - set(behavior_samples.columns)
        if missing:
            raise ValueError(f"Missing required columns in behavior samples: {missing}")

        behavior_samples = behavior_samples.copy()
        behavior_samples["user_id"] = pd.to_numeric(
            behavior_samples["user_id"], errors="raise"
        ).astype("int64")
        behavior_samples["item_id"] = pd.to_numeric(
            behavior_samples["item_id"], errors="raise"
        ).astype("int64")
        behavior_samples["watch_ratio"] = (
            pd.to_numeric(behavior_samples["watch_ratio"], errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )
        behavior_samples["is_positive"] = (
            pd.to_numeric(behavior_samples["is_positive"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

        return behavior_samples

    def _build_user_histories(
        self,
        behavior_samples: pd.DataFrame,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """按用户构造历史序列，并优先保留 watch_ratio 更高的历史 item。"""
        history_samples = behavior_samples[
            behavior_samples["watch_ratio"] >= self.history_min_watch_ratio
        ].copy()

        user_histories: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for user_id, group in history_samples.groupby("user_id", sort=False):
            history_item_indices = group["item_index"].to_numpy(dtype=np.int64)
            watch_ratios = group["watch_ratio"].to_numpy(dtype=np.float32)

            # 按 watch_ratio 降序截取历史，让高观看比例 item 更容易进入 max_history_len。
            interest_order = np.argsort(-watch_ratios)
            user_histories[int(user_id)] = (
                history_item_indices[interest_order],
                watch_ratios[interest_order],
            )

        return user_histories

    def _build_target_samples(
        self,
        behavior_samples: pd.DataFrame,
        max_samples: int | None,
        seed: int,
    ) -> pd.DataFrame:
        """筛选可作为正样本的 target，并确保 target 不会泄漏到用户历史里。"""
        target_samples = behavior_samples.copy()

        if self.only_positive_targets:
            target_samples = target_samples[target_samples["is_positive"].astype(int) == 1].copy()

        # 用户至少要有两条可用历史，排除当前 target 后才可能剩下正向的用户兴趣输入。
        valid_user_ids = {
            user_id
            for user_id, (history_item_indices, _) in self.user_histories.items()
            if len(history_item_indices) >= 2
        }
        target_samples = target_samples[target_samples["user_id"].isin(valid_user_ids)].copy()

        # 进一步排除“用户历史全是同一个 target item”的样本，避免 __getitem__ 里回退到泄漏历史。
        has_non_target_history = [
            np.any(self.user_histories[int(user_id)][0] != int(target_item_index))
            for user_id, target_item_index in zip(
                target_samples["user_id"].to_numpy(),
                target_samples["item_index"].to_numpy(),
                strict=False,
            )
        ]
        target_samples = target_samples.loc[has_non_target_history].copy()

        if max_samples is not None and len(target_samples) > max_samples:
            target_samples = target_samples.sample(
                n=max_samples,
                random_state=seed,
                replace=False,
            ).reset_index(drop=True)
        else:
            target_samples = target_samples.reset_index(drop=True)

        return target_samples

    def __len__(self) -> int:
        return len(self.target_user_ids)

    def __getitem__(self, idx: int):
        user_id = int(self.target_user_ids[idx])
        target_item_id = int(self.target_item_ids[idx])
        target_item_index = int(self.target_item_indices[idx])
        target_watch_ratio = float(self.target_watch_ratios[idx])

        history_item_indices, watch_ratios = self.user_histories[user_id]

        # 防止 target 泄漏：当前正样本 item 不允许同时出现在用户历史输入中。
        keep = history_item_indices != target_item_index
        history_item_indices = history_item_indices[keep]
        watch_ratios = watch_ratios[keep]

        if len(history_item_indices) == 0:
            raise RuntimeError(
                f"User {user_id} has no non-target history for item {target_item_id}."
            )

        # 截断到模型允许的最大历史长度；padding 部分会通过 mask 在 UserEncoder 中屏蔽。
        history_item_indices = history_item_indices[: self.max_history_len]
        watch_ratios = watch_ratios[: self.max_history_len]

        seq_len = len(history_item_indices)

        padded_indices = np.zeros(self.max_history_len, dtype=np.int64)
        padded_watch_ratios = np.zeros(self.max_history_len, dtype=np.float32)
        mask = np.zeros(self.max_history_len, dtype=np.float32)

        padded_indices[:seq_len] = history_item_indices
        padded_watch_ratios[:seq_len] = watch_ratios
        mask[:seq_len] = 1.0

        # dataset 只提供原始 BGE 文本向量，行为空间适配交给共享 ItemEncoder 完成。
        history_item_text_embs = self.item_text_embeddings[padded_indices]
        target_item_text_emb = self.item_text_embeddings[target_item_index]

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "target_item_id": torch.tensor(target_item_id, dtype=torch.long),
            "target_watch_ratio": torch.tensor(target_watch_ratio, dtype=torch.float32),
            "history_item_text_embs": torch.from_numpy(history_item_text_embs),
            "watch_ratios": torch.from_numpy(padded_watch_ratios),
            "mask": torch.from_numpy(mask),
            "target_item_text_emb": torch.from_numpy(target_item_text_emb),
        }
