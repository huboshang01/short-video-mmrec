from pathlib import Path
import argparse
import json

import faiss
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = PROJECT_ROOT / "outputs" / "embeddings"
INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 命令行参数：指定查询视频、返回数量，以及索引/向量/元数据文件位置
def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve semantically similar videos by video_id.")

    parser.add_argument("--video-id", type=int, required=True, help="Query video_id.")
    parser.add_argument("--topk", type=int, default=10, help="Number of similar videos to return.")
    parser.add_argument("--index-path", type=str, default=str(INDEX_DIR / "video_text_faiss.index"), help="Path to FAISS index.")
    parser.add_argument("--embeddings", type=str, default=str(EMB_DIR / "video_text_embeddings.npy"), help="Path to video text embeddings.")
    parser.add_argument("--video-ids", type=str, default=str(EMB_DIR / "video_ids.npy"), help="Path to video ids.")
    parser.add_argument("--meta", type=str, default=str(EMB_DIR / "video_text_meta.csv"), help="Path to video text metadata.")
    parser.add_argument("--config", type=str, default=str(INDEX_DIR / "faiss_index_config.json"), help="Path to FAISS index config.")
    parser.add_argument("--save", action="store_true", help="Save retrieval result to outputs/reports.")

    args = parser.parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive.")
    return args


# 资源加载：读取 FAISS 索引、原始向量、video_id 映射、文本元数据和索引配置
def load_resources(args) -> tuple:
    index_path = Path(args.index_path)
    emb_path = Path(args.embeddings)
    ids_path = Path(args.video_ids)
    meta_path = Path(args.meta)
    config_path = Path(args.config)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")

    if not ids_path.exists():
        raise FileNotFoundError(f"Video ids not found: {ids_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    index = faiss.read_index(str(index_path))
    embeddings = np.load(emb_path).astype("float32")
    video_ids = np.load(ids_path).astype("int64")
    embeddings = np.ascontiguousarray(embeddings)
    video_ids = np.ascontiguousarray(video_ids)

    meta_df = pd.read_csv(
        meta_path,
        encoding="utf-8-sig",
        lineterminator="\n",
        low_memory=False,
    )

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

    if video_ids.ndim != 1:
        raise ValueError(f"Video ids must be 1D, got shape {video_ids.shape}")

    if embeddings.shape[0] != video_ids.shape[0]:
        raise ValueError(
            f"Mismatch: embeddings rows={embeddings.shape[0]}, video_ids={video_ids.shape[0]}"
        )

    if index.d != embeddings.shape[1]:
        raise ValueError(f"Mismatch: index dim={index.d}, embedding dim={embeddings.shape[1]}")

    if index.ntotal != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: index.ntotal={index.ntotal}, embeddings rows={embeddings.shape[0]}"
        )

    if np.unique(video_ids).size != video_ids.size:
        raise ValueError("video_ids contains duplicates.")

    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf.")

    if "video_id" not in meta_df.columns or "video_text" not in meta_df.columns:
        raise ValueError("meta file must contain video_id and video_text columns.")

    return index, embeddings, video_ids, meta_df, config


# 查找表构建：建立 video_id 到向量行号、video_id 到文本内容的快速映射
def build_lookup(video_ids: np.ndarray, meta_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, str]]:
    id_to_row = {int(vid): i for i, vid in enumerate(video_ids.tolist())}

    meta_df = meta_df.copy()
    meta_df["video_id"] = pd.to_numeric(meta_df["video_id"], errors="raise").astype("int64")
    meta_df["video_text"] = meta_df["video_text"].fillna("").astype(str)
    meta_df = meta_df.drop_duplicates("video_id")
    id_to_text = dict(zip(meta_df["video_id"], meta_df["video_text"]))

    return id_to_row, id_to_text


# 相似视频检索：取查询视频向量，调用 FAISS search，并过滤掉查询视频自身
def search_similar_videos(
    query_video_id: int,
    topk: int,
    index,
    embeddings: np.ndarray,
    id_to_row: dict,
    id_to_text: dict,
    metric: str,
):
    if query_video_id not in id_to_row:
        raise ValueError(f"video_id={query_video_id} not found in video_ids.npy")

    query_row = id_to_row[query_video_id]
    query_vector = embeddings[query_row:query_row + 1].copy().astype("float32")

    # 如果索引用的是 cosine，则查询向量也必须 L2 normalize。
    # Step 4 中 cosine = normalize + IndexFlatIP。
    if metric == "cosine":
        faiss.normalize_L2(query_vector)

    # 多取 1 个，因为 Top-1 通常是它自己，需要过滤掉自己
    search_k = min(topk + 1, index.ntotal)
    scores, retrieved_ids = index.search(query_vector, search_k)

    results = []
    for score, vid in zip(scores[0], retrieved_ids[0]):
        vid = int(vid)

        # IndexIDMap 在没有足够结果时可能返回 -1
        if vid == -1:
            continue

        # 过滤自己
        if vid == query_video_id:
            continue

        results.append(
            {
                "rank": len(results) + 1,
                "video_id": vid,
                "score": float(score),
                "video_text": id_to_text.get(vid, ""),
            }
        )

        if len(results) >= topk:
            break

    query_info = {
        "video_id": query_video_id,
        "video_text": id_to_text.get(query_video_id, ""),
    }

    return query_info, pd.DataFrame(results)


# 结果展示：在终端中打印查询视频及其相似视频
def print_results(query_info: dict, result_df: pd.DataFrame):
    print("=" * 100)
    print("[QUERY VIDEO]")
    print("video_id:", query_info["video_id"])
    print("video_text:")
    print(query_info["video_text"][:800])

    print("=" * 100)
    print("[SIMILAR VIDEOS]")

    if result_df.empty:
        print("No results found.")
        return

    for _, row in result_df.iterrows():
        print("-" * 100)
        print(f"rank: {int(row['rank'])}")
        print(f"video_id: {int(row['video_id'])}")
        print(f"score: {row['score']:.6f}")
        print("video_text:")
        print(str(row["video_text"])[:800])


# 结果保存：可选地把相似视频列表保存为 CSV，便于后续查看和汇报
def save_results(query_video_id: int, topk: int, result_df: pd.DataFrame) -> Path:
    out_path = REPORT_DIR / f"similar_video_{query_video_id}_top{topk}.csv"
    save_df = result_df.copy()
    save_df.insert(0, "query_video_id", query_video_id)
    save_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main():
    args = parse_args()

    print("=" * 100)
    print("Similar Video Retrieval - KuaiRec V1")
    print("=" * 100)
    print(f"[INFO] query video_id: {args.video_id}")
    print(f"[INFO] topk: {args.topk}")

    index, embeddings, video_ids, meta_df, config = load_resources(args)
    id_to_row, id_to_text = build_lookup(video_ids, meta_df)

    metric = config.get("metric", "cosine")
    if metric not in {"cosine", "ip", "l2"}:
        raise ValueError(f"Unsupported index metric: {metric}")

    print(f"[INFO] index metric: {metric}")
    print(f"[INFO] index.ntotal: {index.ntotal}")
    print(f"[INFO] embeddings shape: {embeddings.shape}")

    query_info, result_df = search_similar_videos(
        query_video_id=args.video_id,
        topk=args.topk,
        index=index,
        embeddings=embeddings,
        id_to_row=id_to_row,
        id_to_text=id_to_text,
        metric=metric,
    )

    print_results(query_info, result_df)

    if args.save:
        out_path = save_results(args.video_id, args.topk, result_df)
        print("=" * 100)
        print(f"[SAVED] {out_path}")

    print("=" * 100)
    print("Similar video retrieval done.")


if __name__ == "__main__":
    main()
