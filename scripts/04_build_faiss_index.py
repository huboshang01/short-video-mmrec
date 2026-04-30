from pathlib import Path
import argparse
import json

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = PROJECT_ROOT / "outputs" / "embeddings"
INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# 命令行参数：控制输入向量、ID 映射、相似度方式和索引输出位置
def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index for KuaiRec video text embeddings.")

    parser.add_argument("--embeddings", type=str, default=str(EMB_DIR / "video_text_embeddings.npy"), help="Path to video text embeddings .npy file.")
    parser.add_argument("--video-ids", type=str, default=str(EMB_DIR / "video_ids.npy"), help="Path to video ids .npy file.")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "ip", "l2"], help="Similarity metric. cosine uses inner product on L2-normalized vectors.")
    parser.add_argument("--index-path", type=str, default=str(INDEX_DIR / "video_text_faiss.index"), help="Output FAISS index path.")
    parser.add_argument("--check-topk", type=int, default=5, help="Top-k size used by the small self-retrieval check.")

    args = parser.parse_args()
    if args.check_topk <= 0:
        raise ValueError("--check-topk must be positive.")
    return args


# 路径展示：项目内路径保存为相对路径，项目外路径保留绝对路径
def format_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# 输入加载：读取文本向量和 video_id，并检查行数、维度、重复 ID 和非法数值
def load_embeddings(emb_path: Path, ids_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")

    if not ids_path.exists():
        raise FileNotFoundError(f"Video ids file not found: {ids_path}")

    embeddings = np.load(emb_path)
    video_ids = np.load(ids_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError(f"Embeddings cannot be empty, got shape {embeddings.shape}")

    if video_ids.ndim != 1:
        raise ValueError(f"Video ids must be 1D, got shape {video_ids.shape}")

    if embeddings.shape[0] != video_ids.shape[0]:
        raise ValueError(
            f"Row mismatch: embeddings rows={embeddings.shape[0]}, video_ids={video_ids.shape[0]}"
        )

    if np.unique(video_ids).size != video_ids.size:
        raise ValueError("video_ids contains duplicates.")

    embeddings = embeddings.astype("float32")
    embeddings = np.ascontiguousarray(embeddings)

    video_ids = video_ids.astype("int64")
    video_ids = np.ascontiguousarray(video_ids)

    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf.")

    return embeddings, video_ids


# 索引构建：小规模 V1 视频库使用精确 Flat 索引，外层 IDMap 让结果直接返回 video_id
def build_index(embeddings: np.ndarray, video_ids: np.ndarray, metric: str):
    num_vectors, dim = embeddings.shape
    vectors = embeddings.copy()

    if metric == "cosine":
        # cosine = 先做 L2 归一化，再用内积检索
        faiss.normalize_L2(vectors)
        base_index = faiss.IndexFlatIP(dim)
    elif metric == "ip":
        base_index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        base_index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    index = faiss.IndexIDMap2(base_index)
    index.add_with_ids(vectors, video_ids)

    if index.ntotal != num_vectors:
        raise RuntimeError(f"Index size mismatch: index.ntotal={index.ntotal}, num_vectors={num_vectors}")

    return index, vectors


# 自检：用前几条向量检索自身，正常情况下 Top-1 应该等于自己的 video_id
def sanity_check(index, vectors: np.ndarray, video_ids: np.ndarray, topk: int) -> None:
    print("=" * 80)
    print("[SANITY CHECK]")

    query_count = min(3, vectors.shape[0])
    topk = min(topk, vectors.shape[0])
    query = vectors[:query_count]
    scores, retrieved_ids = index.search(query, topk)

    for i in range(query.shape[0]):
        print("-" * 80)
        print("query video_id:", int(video_ids[i]))
        print("retrieved ids:", retrieved_ids[i].tolist())
        print("scores:", scores[i].tolist())

        if int(retrieved_ids[i][0]) != int(video_ids[i]):
            print("[WARN] Top-1 is not itself. Please check embedding/index alignment.")


# 配置保存：记录索引构建参数，方便后续检索脚本确认向量和索引是否匹配
def save_config(
    config_path: Path,
    index_path: Path,
    emb_path: Path,
    ids_path: Path,
    index,
    embeddings: np.ndarray,
    metric: str,
) -> None:
    config = {
        "metric": metric,
        "score_order": "higher_is_better" if metric in {"cosine", "ip"} else "lower_is_better",
        "index_type": type(index).__name__,
        "base_index": "IndexFlatIP" if metric in {"cosine", "ip"} else "IndexFlatL2",
        "num_vectors": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "index_path": format_path(index_path),
        "embeddings_path": format_path(emb_path),
        "video_ids_path": format_path(ids_path),
        "note": "IndexIDMap2 makes FAISS return original video_id values. For metric=cosine, vectors are L2-normalized before adding to IndexFlatIP.",
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    emb_path = Path(args.embeddings)
    ids_path = Path(args.video_ids)
    index_path = Path(args.index_path)

    print("=" * 80)
    print("Build FAISS Index for KuaiRec V1")
    print("=" * 80)
    print(f"[INFO] embeddings: {emb_path}")
    print(f"[INFO] video ids: {ids_path}")
    print(f"[INFO] metric: {args.metric}")
    print(f"[INFO] output index: {index_path}")
    print(f"[INFO] check topk: {args.check_topk}")

    embeddings, video_ids = load_embeddings(emb_path, ids_path)

    print("=" * 80)
    print("[DATA]")
    print("embeddings shape:", embeddings.shape)
    print("embeddings dtype:", embeddings.dtype)
    print("video_ids shape:", video_ids.shape)
    print("first video ids:", video_ids[:10].tolist())

    norms = np.linalg.norm(embeddings, axis=1)
    print("embedding norm min:", float(norms.min()))
    print("embedding norm max:", float(norms.max()))
    print("embedding norm mean:", float(norms.mean()))

    index, vectors_for_check = build_index(embeddings, video_ids, args.metric)

    print("=" * 80)
    print("[INDEX]")
    print("index type:", type(index).__name__)
    print("index.ntotal:", index.ntotal)
    print("dimension:", embeddings.shape[1])

    sanity_check(index, vectors_for_check, video_ids, topk=args.check_topk)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    config_path = INDEX_DIR / "faiss_index_config.json"
    save_config(config_path, index_path, emb_path, ids_path, index, embeddings, args.metric)

    print("=" * 80)
    print("[SAVED]")
    print(f"index: {index_path}")
    print(f"config: {config_path}")
    print("=" * 80)
    print("Build FAISS index done.")


if __name__ == "__main__":
    main()
