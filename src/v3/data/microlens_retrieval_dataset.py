"""
V3 MicroLens-100K 召回 Dataset 与多模态特征缓存。

本模块负责把 processed CSV 和官方预提取特征组装成训练样本：
    1. Dataset 只返回 item index，不直接把大特征塞进 DataLoader batch。
    2. MultimodalFeatureCache 使用 numpy mmap 读取 text/image/video 特征。
    3. 负样本在 __getitem__ 中从全量 item 采样，并避开该用户已交互 item。

这种设计能让训练循环统一按 item index 拉取特征，后续切换 text-only / visual-only
或 hard negative 时不需要改 CSV 样本格式。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class UserHistory:
    """单个用户按时间排序后的历史 item 序列。"""

    item_indices: np.ndarray
    sort_keys: np.ndarray


@dataclass
class MultimodalFeatureCache:
    """官方 text/image/video item 特征缓存，按 item_index 直接取行。"""

    text: np.ndarray
    image: np.ndarray
    video: np.ndarray

    @classmethod
    def from_config(cls, feature_config_path: str | Path) -> "MultimodalFeatureCache":
        config_path = resolve_project_path(feature_config_path)
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        features = config["features"]
        return cls(
            text=np.load(resolve_project_path(features["text"]["npy_path"]), mmap_mode="r"),
            image=np.load(resolve_project_path(features["image"]["npy_path"]), mmap_mode="r"),
            video=np.load(resolve_project_path(features["video"]["npy_path"]), mmap_mode="r"),
        )

    @property
    def num_items(self) -> int:
        return int(self.text.shape[0])

    @property
    def text_dim(self) -> int:
        return int(self.text.shape[1])

    @property
    def image_dim(self) -> int:
        return int(self.image.shape[1])

    @property
    def video_dim(self) -> int:
        return int(self.video.shape[1])

    def validate(self) -> None:
        """检查三类特征行数一致，避免 item_index 取错模态行。"""
        if self.text.ndim != 2 or self.image.ndim != 2 or self.video.ndim != 2:
            raise ValueError("All multimodal features must be 2D arrays.")
        if not (self.text.shape[0] == self.image.shape[0] == self.video.shape[0]):
            raise ValueError(
                "Feature row count mismatch: "
                f"text={self.text.shape}, image={self.image.shape}, video={self.video.shape}"
            )

    def gather(self, item_indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        """按任意形状的 item_indices 取特征，并恢复相同前缀维度。"""
        original_shape = tuple(item_indices.shape)
        flat_indices = item_indices.detach().cpu().numpy().reshape(-1)

        # memmap 切片后 copy，避免 torch 对只读 numpy buffer 发 warning。
        text = torch.from_numpy(np.asarray(self.text[flat_indices]).copy())
        image = torch.from_numpy(np.asarray(self.image[flat_indices]).copy())
        video = torch.from_numpy(np.asarray(self.video[flat_indices]).copy())

        return {
            "text_features": text.reshape(*original_shape, self.text_dim).to(device=device, dtype=torch.float32),
            "image_features": image.reshape(*original_shape, self.image_dim).to(device=device, dtype=torch.float32),
            "video_features": video.reshape(*original_shape, self.video_dim).to(device=device, dtype=torch.float32),
        }


class MicroLensRetrievalDataset(Dataset):
    """
    MicroLens 隐式反馈召回训练 Dataset。

    单条样本：
        history_item_indices: target 之前最近 max_history_len 个历史 item。
        target_item_index: 当前正样本 item。
        negative_item_indices: 从全量 item 中采样的未交互负样本。
    """

    def __init__(
        self,
        target_samples_path: str | Path,
        item_ids_path: str | Path,
        history_samples_path: str | Path | None = None,
        max_history_len: int = 50,
        num_negatives: int = 32,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if max_history_len <= 0:
            raise ValueError("max_history_len must be positive.")
        if num_negatives <= 0:
            raise ValueError("num_negatives must be positive.")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None.")

        self.target_samples_path = resolve_project_path(target_samples_path)
        self.history_samples_path = resolve_project_path(history_samples_path or target_samples_path)
        self.max_history_len = int(max_history_len)
        self.num_negatives = int(num_negatives)
        self.seed = int(seed)

        self.item_ids = load_item_ids(resolve_project_path(item_ids_path))
        self.item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(self.item_ids)}
        self.all_item_indices = np.arange(len(self.item_ids), dtype=np.int64)

        history_rows = load_positive_rows(self.history_samples_path, self.item_id_to_index)
        target_rows = load_positive_rows(self.target_samples_path, self.item_id_to_index)
        self.user_histories = build_user_histories(history_rows)
        self.user_seen_items = build_user_seen_items(history_rows + target_rows)
        self.targets = self._filter_targets_with_history(target_rows)

        if max_samples is not None and len(self.targets) > max_samples:
            rng = np.random.default_rng(self.seed)
            chosen = rng.choice(len(self.targets), size=max_samples, replace=False)
            self.targets = [self.targets[int(idx)] for idx in sorted(chosen.tolist())]

        if not self.targets:
            raise ValueError("No target samples with causal history. Please check split/history paths.")

    def _filter_targets_with_history(self, target_rows: list[dict[str, int]]) -> list[dict[str, int]]:
        """只保留 target 时间之前已经有历史行为的样本。"""
        valid_targets: list[dict[str, int]] = []
        for row in target_rows:
            history = self.user_histories.get(int(row["user_id"]))
            if history is None:
                continue
            if bisect_left(history.sort_keys, int(row["sort_key"])) > 0:
                valid_targets.append(row)
        return valid_targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        target = self.targets[idx]
        user_id = int(target["user_id"])
        target_item_index = int(target["item_index"])
        target_sort_key = int(target["sort_key"])

        history = self.user_histories[user_id]
        history_end = bisect_left(history.sort_keys, target_sort_key)
        history_start = max(0, history_end - self.max_history_len)
        selected = history.item_indices[history_start:history_end]

        history_item_indices = np.zeros(self.max_history_len, dtype=np.int64)
        history_mask = np.zeros(self.max_history_len, dtype=np.float32)
        seq_len = len(selected)
        history_item_indices[:seq_len] = selected
        history_mask[:seq_len] = 1.0

        negatives = self._sample_negatives(user_id=user_id, target_item_index=target_item_index, idx=idx)

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "history_item_indices": torch.from_numpy(history_item_indices),
            "history_mask": torch.from_numpy(history_mask),
            "target_item_index": torch.tensor(target_item_index, dtype=torch.long),
            "negative_item_indices": torch.from_numpy(negatives),
        }

    def _sample_negatives(self, user_id: int, target_item_index: int, idx: int) -> np.ndarray:
        """从未交互 item 中采样负样本；对 idx 加 seed 保证验证集可复现。"""
        rng = np.random.default_rng(self.seed + idx)
        forbidden = set(self.user_seen_items.get(user_id, set()))
        forbidden.add(int(target_item_index))

        negatives: list[int] = []
        used = set(forbidden)
        max_attempts = self.num_negatives * 50
        for _ in range(max_attempts):
            candidate = int(rng.integers(0, len(self.all_item_indices)))
            if candidate in used:
                continue
            negatives.append(candidate)
            used.add(candidate)
            if len(negatives) == self.num_negatives:
                break

        if len(negatives) < self.num_negatives:
            candidates = [int(item) for item in self.all_item_indices if int(item) not in used]
            if len(candidates) < self.num_negatives - len(negatives):
                raise ValueError(f"Not enough negative candidates for user_id={user_id}.")
            negatives.extend(rng.choice(candidates, size=self.num_negatives - len(negatives), replace=False).tolist())

        return np.asarray(negatives, dtype=np.int64)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_item_ids(path: Path) -> list[int]:
    """读取 item_index,item_id 映射，并检查行号连续。"""
    item_ids: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for expected_index, row in enumerate(reader):
            item_index = int(row["item_index"])
            item_id = int(row["item_id"])
            if item_index != expected_index:
                raise ValueError(f"item_index must be contiguous: expected={expected_index}, got={item_index}")
            item_ids.append(item_id)
    if not item_ids:
        raise ValueError(f"No item ids loaded from {path}")
    return item_ids


def load_positive_rows(path: Path, item_id_to_index: dict[int, int]) -> list[dict[str, int]]:
    """读取 label=1 的隐式正反馈样本，并附上 item_index。"""
    rows: list[dict[str, int]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"user_id", "item_id", "sort_key", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        for row in reader:
            if int(float(row["label"])) != 1:
                continue
            item_id = int(row["item_id"])
            if item_id not in item_id_to_index:
                continue
            rows.append(
                {
                    "user_id": int(row["user_id"]),
                    "item_id": item_id,
                    "item_index": int(item_id_to_index[item_id]),
                    "sort_key": int(row["sort_key"]),
                }
            )

    if not rows:
        raise ValueError(f"No positive rows loaded from {path}")
    return rows


def build_user_histories(rows: list[dict[str, int]]) -> dict[int, UserHistory]:
    """将正反馈样本按用户聚合成时间序列。"""
    grouped: dict[int, list[dict[str, int]]] = {}
    for row in rows:
        grouped.setdefault(int(row["user_id"]), []).append(row)

    histories: dict[int, UserHistory] = {}
    for user_id, user_rows in grouped.items():
        user_rows.sort(key=lambda row: (int(row["sort_key"]), int(row["item_index"])))
        histories[user_id] = UserHistory(
            item_indices=np.asarray([row["item_index"] for row in user_rows], dtype=np.int64),
            sort_keys=np.asarray([row["sort_key"] for row in user_rows], dtype=np.int64),
        )
    return histories


def build_user_seen_items(rows: list[dict[str, int]]) -> dict[int, set[int]]:
    """记录每个用户已交互 item，负采样时避开这些 item。"""
    seen: dict[int, set[int]] = {}
    for row in rows:
        seen.setdefault(int(row["user_id"]), set()).add(int(row["item_index"]))
    return seen
