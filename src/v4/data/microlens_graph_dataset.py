"""
V4 MicroLens-100K LightGCN 图数据工具。

本模块负责把 V3 已处理好的 MicroLens 行为 CSV 转成 V4 协同过滤所需的数据形式：
    1. 读取 item_id -> item_index 映射，保证和 V3 多模态特征行号一致；
    2. 读取 label=1 的隐式正反馈样本；
    3. 构建 raw user_id -> 连续 user_index 映射；
    4. 将正反馈转换成 LightGCN 的 user-item 图边；
    5. 为 BPR 训练动态采样未交互负 item。

训练阶段用 train split 构图和采样 BPR 三元组；评估阶段复用 train 用户映射，
只给 train 中有历史的用户做 full-catalog ranking。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Interaction:
    """一条已经完成连续编号映射的 user-item 正反馈边。"""

    # user_index/item_index 都是从 0 开始的连续编号，适合直接喂给 embedding。
    user_index: int
    item_index: int


def resolve_project_path(path: str | Path) -> Path:
    """支持传入项目相对路径或绝对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_item_ids(path: str | Path) -> list[int]:
    """读取 V3 生成的连续 item_index,item_id 映射。"""
    resolved = resolve_project_path(path)
    item_ids: list[int] = []
    with resolved.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"item_index", "item_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{resolved} missing columns: {sorted(missing)}")

        for expected_index, row in enumerate(reader):
            item_index = int(row["item_index"])
            item_id = int(row["item_id"])
            # item_index 必须严格连续，否则 embedding 行号会和行为样本对不上。
            if item_index != expected_index:
                raise ValueError(f"item_index must be contiguous: expected={expected_index}, got={item_index}")
            item_ids.append(item_id)

    if not item_ids:
        raise ValueError(f"No item ids loaded from {resolved}")
    return item_ids


def load_positive_rows(path: str | Path, item_id_to_index: dict[int, int]) -> list[dict[str, int]]:
    """读取 label=1 的正反馈行，并把原始 item_id 转成连续 item_index。"""
    resolved = resolve_project_path(path)
    rows: list[dict[str, int]] = []
    with resolved.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"user_id", "item_id", "sort_key", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{resolved} missing columns: {sorted(missing)}")

        for row in reader:
            # MicroLens 处理后的样本都是隐式反馈；这里仍保留 label 判断，便于兼容后续扩展。
            if int(float(row["label"])) != 1:
                continue
            item_id = int(row["item_id"])
            # 如果某个 item 没有进入 item_ids 映射，直接跳过，避免越界访问 embedding。
            if item_id not in item_id_to_index:
                continue
            rows.append(
                {
                    "user_id": int(row["user_id"]),
                    "item_index": int(item_id_to_index[item_id]),
                    "sort_key": int(row["sort_key"]),
                }
            )

    if not rows:
        raise ValueError(f"No positive rows loaded from {resolved}")
    # 固定排序让用户映射、抽样和调试输出更可复现。
    rows.sort(key=lambda row: (int(row["user_id"]), int(row["sort_key"]), int(row["item_index"])))
    return rows


def build_user_mapping(rows: list[dict[str, int]]) -> dict[int, int]:
    """构建稳定的 raw user_id -> 连续 user_index 映射。"""
    # LightGCN 的 user embedding 需要连续行号；按 raw user_id 排序保证每次映射一致。
    user_ids = sorted({int(row["user_id"]) for row in rows})
    if not user_ids:
        raise ValueError("Cannot build user mapping from empty rows.")
    return {user_id: idx for idx, user_id in enumerate(user_ids)}


def invert_mapping(mapping: dict[int, int]) -> list[int]:
    """把 raw_id -> index 反转成按 index 排列的 raw_id 列表。"""
    ordered = [0] * len(mapping)
    for raw_id, index in mapping.items():
        ordered[int(index)] = int(raw_id)
    return ordered


def rows_to_interactions(rows: list[dict[str, int]], user_id_to_index: dict[int, int]) -> list[Interaction]:
    """将行为行转换成连续编号后的图边，自动丢弃 train 中未见过的用户。"""
    interactions: list[Interaction] = []
    for row in rows:
        user_id = int(row["user_id"])
        # val/test Dataset 会复用 train 的 user 映射；冷启动用户没有 LightGCN 表示，跳过。
        if user_id not in user_id_to_index:
            continue
        interactions.append(
            Interaction(
                user_index=int(user_id_to_index[user_id]),
                item_index=int(row["item_index"]),
            )
        )
    return interactions


def build_user_seen_items(interactions: list[Interaction]) -> dict[int, set[int]]:
    """构建每个用户已交互 item 集合，用于负采样和评估过滤。"""
    seen: dict[int, set[int]] = {}
    for interaction in interactions:
        seen.setdefault(int(interaction.user_index), set()).add(int(interaction.item_index))
    return seen


def build_eval_relevance(
    rows: list[dict[str, int]],
    user_id_to_index: dict[int, int],
) -> dict[int, set[int]]:
    """把 val/test 正反馈聚合成 user_index -> relevant item_index 集合。"""
    relevance: dict[int, set[int]] = {}
    for row in rows:
        user_id = int(row["user_id"])
        # 只保留训练图中出现过的用户，LightGCN 无法为全新用户做协同过滤召回。
        if user_id not in user_id_to_index:
            continue
        user_index = int(user_id_to_index[user_id])
        relevance.setdefault(user_index, set()).add(int(row["item_index"]))
    return relevance


def maybe_sample_interactions(
    interactions: list[Interaction],
    max_samples: int | None,
    seed: int,
) -> list[Interaction]:
    """可选抽样目标交互，但不改变完整图边集合。"""
    if max_samples is None or int(max_samples) == -1 or len(interactions) <= int(max_samples):
        return list(interactions)
    if int(max_samples) <= 0:
        raise ValueError("max_samples must be positive, -1, or None.")

    # 仅抽样训练/验证的 target triples；LightGCN 图仍使用 dataset.interactions 的完整边。
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(interactions), size=int(max_samples), replace=False)
    return [interactions[int(idx)] for idx in sorted(chosen.tolist())]


class LightGCNInteractionDataset(Dataset):
    """
    MicroLens 隐式反馈上的 LightGCN BPR 三元组 Dataset。

    每条样本返回：
        user_index
        positive_item_index
        negative_item_index

    inter   ····················actions 保存完整正反馈边，用于构造 LightGCN train graph；
    targets 是本轮训练/验证会遍历的正样本集合，可通过 max_samples 缩小做 smoke test。
    """

    def __init__(
        self,
        target_samples_path: str | Path,
        item_ids_path: str | Path,
        user_id_to_index: dict[int, int] | None = None,
        user_seen_items: dict[int, set[int]] | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.seed = int(seed)
        # epoch 会参与负采样随机种子，让不同 epoch 的负样本不同但仍可复现。
        self.epoch = 0

        # item 映射沿用 V3，保证 V4 item_index 可与 V3 多模态 item 表征对齐。
        self.item_ids = load_item_ids(item_ids_path)
        self.item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(self.item_ids)}

        rows = load_positive_rows(target_samples_path, self.item_id_to_index)
        # train Dataset 自建用户映射；val/test Dataset 需要传入 train 的用户映射。
        self.user_id_to_index = user_id_to_index or build_user_mapping(rows)
        self.user_ids = invert_mapping(self.user_id_to_index)
        self.interactions = rows_to_interactions(rows, self.user_id_to_index)
        if not self.interactions:
            raise ValueError("No interactions left after applying user mapping.")

        # user_seen_items 优先继承 train 已交互集合，再合并当前 split 的正反馈集合。
        # 训练时它用于负采样避开已交互 item；评估时它用于过滤 train 已看 item。
        self.user_seen_items = {int(user): set(items) for user, items in (user_seen_items or {}).items()}
        for user_index, items in build_user_seen_items(self.interactions).items():
            self.user_seen_items.setdefault(int(user_index), set()).update(items)
        # targets 可小于 interactions，便于快速实验，但不影响图构建所需的完整边集合。
        self.targets = maybe_sample_interactions(self.interactions, max_samples=max_samples, seed=self.seed)
        if not self.targets:
            raise ValueError("No target interactions available for LightGCN training/eval.")

    @property
    def num_users(self) -> int:
        """连续 user_index 总数，也是 user embedding 表大小。"""
        return len(self.user_ids)

    @property
    def num_items(self) -> int:
        """连续 item_index 总数，也是 item embedding 表大小。"""
        return len(self.item_ids)

    def set_epoch(self, epoch: int) -> None:
        """更新 epoch，使确定性负采样在不同 epoch 产生不同负样本。"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """返回本次训练/验证实际遍历的 target 正样本数。"""
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """根据一个正反馈 target 构造 BPR 训练三元组。"""
        interaction = self.targets[idx]
        user_index = int(interaction.user_index)
        positive_item_index = int(interaction.item_index)
        # 负样本从该用户未交互 item 中采样。
        negative_item_index = self._sample_negative(user_index, positive_item_index, idx)

        return {
            "user_index": torch.tensor(user_index, dtype=torch.long),
            "positive_item_index": torch.tensor(positive_item_index, dtype=torch.long),
            "negative_item_index": torch.tensor(negative_item_index, dtype=torch.long),
        }

    def _sample_negative(self, user_index: int, positive_item_index: int, idx: int) -> int:
        """为指定用户采样一个未交互负 item。"""
        # forbidden 包含该用户所有已交互 item，再显式加入当前正样本。
        forbidden = set(self.user_seen_items.get(int(user_index), set()))
        forbidden.add(int(positive_item_index))
        if len(forbidden) >= self.num_items:
            raise ValueError(f"User {user_index} has no negative candidates.")

        # seed + epoch + idx 让采样可复现，同时不同 epoch/样本位置有不同随机流。
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(idx))
        for _ in range(200):
            candidate = int(rng.integers(0, self.num_items))
            if candidate not in forbidden:
                return candidate

        # 如果随机尝试多次都撞到已交互 item，则退化为从完整候选集合中过滤后再采样。
        candidates = np.asarray([idx for idx in range(self.num_items) if idx not in forbidden], dtype=np.int64)
        return int(rng.choice(candidates))
