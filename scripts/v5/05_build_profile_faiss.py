"""
V5 Step 05: 构建 Profile FAISS 召回索引。

这一步和V1 的04_build_faiss_index.py一致，但V5 索引的是 profile embedding。

输入：
    outputs/v5/microlens_100k/embeddings/profile_item_ids.npy
    outputs/v5/microlens_100k/embeddings/profile_embeddings.npy

输出：
    outputs/v5/microlens_100k/indexes/profile_faiss.index
    outputs/v5/microlens_100k/indexes/profile_faiss_config.json

索引使用 IndexIDMap2，检索结果直接返回 item_id，便于案例分析和在线化。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.profile.io import load_yaml
from src.v5.profile.paths import project_relative, resolve_project_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5" / "profile_retrieval.yaml"


def parse_args() -> argparse.Namespace:
    """解析索引构建参数。

    --metric 默认读取 profile_retrieval.yaml；cosine 是当前 profile embedding 的主实验口径。
    """
    parser = argparse.ArgumentParser(description="Build FAISS index for V5 profile embeddings.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--metric", type=str, default="", choices=["", "cosine", "ip", "l2"])
    return parser.parse_args()


def main() -> None:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("FAISS index building requires faiss-cpu and numpy.") from exc

    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    metric = args.metric or cfg["retrieval"]["metric"]

    # Step 04 输出两个并行数组：profile_item_ids[i] 对应 profile_embeddings[i]。
    # 注意这里不是要求 item_id 等于行号，而是要求两个数组长度一致且按相同顺序排列。
    ids_path = resolve_project_path(cfg["data"]["profile_item_ids"])
    emb_path = resolve_project_path(cfg["data"]["profile_embeddings"])
    index_path = resolve_project_path(cfg["data"]["profile_index"])
    config_path = resolve_project_path(cfg["data"]["index_config"])

    item_ids = np.load(ids_path).astype("int64")
    embeddings = np.load(emb_path).astype("float32")
    if item_ids.ndim != 1 or embeddings.ndim != 2 or item_ids.shape[0] != embeddings.shape[0]:
        raise ValueError("profile_item_ids.npy and profile_embeddings.npy are not aligned.")
    if np.unique(item_ids).size != item_ids.size:
        raise ValueError("profile item ids contain duplicates.")

    vectors = np.ascontiguousarray(embeddings.copy())
    if metric == "cosine":
        # FAISS 没有单独的 cosine index；向量 L2 归一化后，内积等价于 cosine similarity。
        faiss.normalize_L2(vectors)
        base = faiss.IndexFlatIP(vectors.shape[1])
    elif metric == "ip":
        base = faiss.IndexFlatIP(vectors.shape[1])
    else:
        base = faiss.IndexFlatL2(vectors.shape[1])

    # IndexIDMap2 把 FAISS 内部行号映射成真实 item_id，后续检索不用再手动反查。
    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vectors, item_ids)

    # 保存 index 以及可复现实验所需的轻量配置。
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": metric,
                "num_vectors": int(index.ntotal),
                "embedding_dim": int(vectors.shape[1]),
                "index_path": project_relative(index_path),
                "embeddings": project_relative(emb_path),
                "item_ids": project_relative(ids_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print("=" * 80)
    print("V5 Step 05: Build Profile FAISS")
    print("=" * 80)
    print(f"index: {project_relative(index_path)}")
    print(f"config: {project_relative(config_path)}")
    print(f"vectors: {index.ntotal}")


if __name__ == "__main__":
    main()
