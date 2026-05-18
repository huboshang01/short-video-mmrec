"""V5 embedding 加载、归一化与内容召回特征构造。"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from src.v5.profile.paths import resolve_project_path


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行做 L2 归一化，使点积可以作为 cosine similarity 使用。"""
    x = x.astype("float32", copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def load_profile_embeddings(
    ids_path: Path,
    emb_path: Path,
    target_item_ids: list[int] | None = None,
) -> tuple[list[int], np.ndarray]:
    """读取 Step 04 生成的 profile embedding，并可按目标 item 子集对齐。

    返回的 item_ids 与 embeddings 是并行数组：item_ids[i] 对应 embeddings[i]。
    """
    item_ids = np.load(ids_path).astype("int64").tolist()
    embeddings = np.load(emb_path).astype("float32")
    if len(item_ids) != embeddings.shape[0]:
        raise ValueError("profile item ids and embeddings row count mismatch.")
    if target_item_ids is None:
        return item_ids, embeddings

    # real item_id -> embedding row，用于把 profile 向量对齐到指定候选 item 顺序。
    row_by_id = {int(item_id): idx for idx, item_id in enumerate(item_ids)}
    keep = [row_by_id[int(item_id)] for item_id in target_item_ids if int(item_id) in row_by_id]
    kept_ids = [int(item_ids[idx]) for idx in keep]
    return kept_ids, embeddings[keep]


def load_feature_config(path: Path) -> dict:
    """读取 V3 产出的官方 text/image/video 特征配置。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_matrix(
    method: str,
    item_ids: list[int],
    feature_config: dict,
    profile_ids: list[int] | None = None,
    profile_embeddings: np.ndarray | None = None,
    fusion_weights: dict | None = None,
) -> tuple[list[int], np.ndarray]:
    """构造 title/profile/multimodal/fusion 内容召回向量。

    method 含义：
        title: 只用官方标题文本向量。
        profile: 只用 V5 MLLM profile 向量。
        multimodal: 拼接官方 text/image/video 特征，不依赖 V3 训练 checkpoint。
        fusion: 拼接 title/profile/image/video，并按配置权重缩放。
    """
    method = method.lower()
    item_ids = [int(x) for x in item_ids]
    row_indices = [item_id - 1 for item_id in item_ids]
    text_path = resolve_project_path(feature_config["features"]["text"]["npy_path"])
    image_path = resolve_project_path(feature_config["features"]["image"]["npy_path"])
    video_path = resolve_project_path(feature_config["features"]["video"]["npy_path"])

    # MicroLens 官方特征按 item_id 升序存储；item_id 从 1 开始，因此行号为 item_id - 1。
    text = np.load(text_path, mmap_mode="r")[row_indices].astype("float32")
    image = np.load(image_path, mmap_mode="r")[row_indices].astype("float32")
    video = np.load(video_path, mmap_mode="r")[row_indices].astype("float32")

    if method == "title":
        # baseline：只看原始标题语义。
        return item_ids, l2_normalize(text)
    if method == "multimodal":
        # V3 风格内容 baseline：官方三模态特征拼接后再整体归一化。
        matrix = np.concatenate([l2_normalize(text), l2_normalize(image), l2_normalize(video)], axis=1)
        return item_ids, l2_normalize(matrix)
    if method in {"profile", "fusion"}:
        if profile_ids is None or profile_embeddings is None:
            raise ValueError("profile ids/embeddings are required.")
        profile_lookup = {int(item_id): idx for idx, item_id in enumerate(profile_ids)}
        keep = [idx for idx, item_id in enumerate(item_ids) if item_id in profile_lookup]
        kept_ids = [item_ids[idx] for idx in keep]
        # profile_item_ids[i] 对应 profile_embeddings[i]；这里按 kept_ids 重排到候选 item 顺序。
        profile = np.stack([profile_embeddings[profile_lookup[item_id]] for item_id in kept_ids]).astype("float32")
        if method == "profile":
            return kept_ids, l2_normalize(profile)

        # 加权拼接不是做分数加权，而是先缩放各路向量，再拼成一个统一召回空间。
        weights = {"title": 0.3, "profile": 0.5, "image": 0.1, "video": 0.1}
        weights.update(fusion_weights or {})
        matrix = np.concatenate(
            [
                l2_normalize(text[keep]) * float(weights["title"]),
                l2_normalize(profile) * float(weights["profile"]),
                l2_normalize(image[keep]) * float(weights["image"]),
                l2_normalize(video[keep]) * float(weights["video"]),
            ],
            axis=1,
        )
        return kept_ids, l2_normalize(matrix)
    raise ValueError(f"Unsupported method: {method}")