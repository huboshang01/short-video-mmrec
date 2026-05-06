from pathlib import Path
import argparse
import json

import faiss
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EMB_DIR = PROJECT_ROOT / "outputs" / "v1" / "embeddings"
INDEX_DIR = PROJECT_ROOT / "outputs" / "v1" / "indexes"
REPORT_DIR = PROJECT_ROOT / "outputs" / "v1" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 命令行参数：控制目标用户、兴趣构建方式、过滤策略和输入输出文件
def parse_args():
    parser = argparse.ArgumentParser(description="User interest recommendation for KuaiRec V1.")

    parser.add_argument("--user-id", type=int, required=True, help="Query user_id.")
    parser.add_argument("--topk", type=int, default=10, help="Number of recommended videos.")
    parser.add_argument("--pos-threshold", type=float, default=1.0, help="Use videos with watch_ratio >= threshold as positive history.")
    parser.add_argument("--max-history", type=int, default=50, help="Maximum number of positive history videos used to build user interest.")
    parser.add_argument("--weight-cap", type=float, default=5.0, help="Cap watch_ratio weights to avoid extreme values dominating the user vector.")
    parser.add_argument("--exclude-mode", type=str, default="profile", choices=["profile", "all_seen", "none"], help="profile: exclude profile videos; all_seen: exclude all watched videos; none: do not exclude any video.")
    parser.add_argument("--candidate-k", type=int, default=500, help="Number of candidates retrieved from FAISS before filtering.")
    parser.add_argument("--save", action="store_true", help="Save recommendation result to outputs/v1/reports.")
    parser.add_argument("--small-matrix", type=str, default="", help="Path to small_matrix.csv. If empty, search under data/raw.")
    parser.add_argument("--index-path", type=str, default=str(INDEX_DIR / "video_text_faiss.index"), help="Path to FAISS index.")
    parser.add_argument("--embeddings", type=str, default=str(EMB_DIR / "video_text_embeddings.npy"), help="Path to video text embeddings.")
    parser.add_argument("--video-ids", type=str, default=str(EMB_DIR / "video_ids.npy"), help="Path to video ids.")
    parser.add_argument("--meta", type=str, default=str(EMB_DIR / "video_text_meta.csv"), help="Path to video text metadata.")
    parser.add_argument("--index-config", type=str, default=str(INDEX_DIR / "faiss_index_config.json"), help="Path to FAISS index config.")

    args = parser.parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive.")
    if args.max_history <= 0:
        raise ValueError("--max-history must be positive.")
    if args.weight_cap <= 0:
        raise ValueError("--weight-cap must be positive.")
    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be positive.")
    return args


# 数据定位：在 data/raw 下递归查找 KuaiRec 原始文件
def find_file(filename: str) -> Path:
    matches = list(RAW_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename} under {RAW_DIR}")
    return matches[0]


# CSV 读取：统一处理 KuaiRec 文本中可能存在的 \r 控制字符
def read_csv_kuairec(path: Path, usecols=None, chunksize: int | None = None):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        lineterminator="\n",
        low_memory=False,
        usecols=usecols,
        chunksize=chunksize,
    )


# JSON 读取：加载 FAISS 建索引阶段保存的配置
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 用户历史读取：分块读取 small_matrix，只保留目标用户，避免全量加载交互表
def load_user_history(small_path: Path, user_id: int, chunksize: int = 500_000) -> pd.DataFrame:
    chunks = []
    for chunk in read_csv_kuairec(
        small_path,
        usecols=["user_id", "video_id", "watch_ratio"],
        chunksize=chunksize,
    ):
        user_chunk = chunk[chunk["user_id"] == user_id]
        if not user_chunk.empty:
            chunks.append(user_chunk)

    if not chunks:
        raise ValueError(f"user_id={user_id} not found in {small_path}")

    user_df = pd.concat(chunks, ignore_index=True)
    user_df["user_id"] = pd.to_numeric(user_df["user_id"], errors="raise").astype("int64")
    user_df["video_id"] = pd.to_numeric(user_df["video_id"], errors="raise").astype("int64")
    user_df["watch_ratio"] = pd.to_numeric(user_df["watch_ratio"], errors="coerce").fillna(0.0).astype("float32")
    return user_df


# 资源加载：读取用户历史、FAISS 索引、item 向量、video_id 映射、文本元数据和索引配置
def load_resources(args):
    small_path = Path(args.small_matrix) if args.small_matrix else find_file("small_matrix.csv")

    index_path = Path(args.index_path)
    emb_path = Path(args.embeddings)
    ids_path = Path(args.video_ids)
    meta_path = Path(args.meta)
    index_config_path = Path(args.index_config)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Video ids not found: {ids_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    print(f"[INFO] reading small_matrix: {small_path}")
    user_df = load_user_history(small_path, args.user_id)

    index = faiss.read_index(str(index_path))
    embeddings = np.load(emb_path).astype("float32")
    video_ids = np.load(ids_path).astype("int64")
    embeddings = np.ascontiguousarray(embeddings)
    video_ids = np.ascontiguousarray(video_ids)

    meta_df = read_csv_kuairec(meta_path)
    index_config = load_json(index_config_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
    if video_ids.ndim != 1:
        raise ValueError(f"Video ids must be 1D, got shape {video_ids.shape}")
    if embeddings.shape[0] != video_ids.shape[0]:
        raise ValueError("embeddings and video_ids row count mismatch.")
    if index.d != embeddings.shape[1]:
        raise ValueError(f"FAISS index dim={index.d}, embedding dim={embeddings.shape[1]}")
    if index.ntotal != embeddings.shape[0]:
        raise ValueError("FAISS index size and embedding rows mismatch.")
    if np.unique(video_ids).size != video_ids.size:
        raise ValueError("video_ids contains duplicates.")
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf.")

    if "video_id" not in meta_df.columns or "video_text" not in meta_df.columns:
        raise ValueError("meta file must contain video_id and video_text columns.")

    return user_df, index, embeddings, video_ids, meta_df, index_config


# 查找表构建：建立 video_id 到向量行号、video_id 到文本内容的快速映射
def build_lookup(video_ids: np.ndarray, meta_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, str]]:
    id_to_row = {int(vid): i for i, vid in enumerate(video_ids.tolist())}

    meta_df = meta_df.copy()
    meta_df["video_id"] = pd.to_numeric(meta_df["video_id"], errors="raise").astype("int64")
    meta_df["video_text"] = meta_df["video_text"].fillna("").astype(str)
    meta_df = meta_df.drop_duplicates("video_id")
    id_to_text = dict(zip(meta_df["video_id"], meta_df["video_text"]))

    return id_to_row, id_to_text


# ※ 用户画像构建：从用户高 watch_ratio 历史中选取正反馈 item，并生成聚合权重。权重就是过滤后的watch_ratio的归一化值
def build_user_profile(
    user_id: int,
    user_df: pd.DataFrame,
    id_to_row: dict,
    max_history: int,
    pos_threshold: float,
    weight_cap: float,
):
    if user_df.empty:
        raise ValueError(f"user_id={user_id} has no interaction history.")

    # 只保留有文本 embedding 的视频
    user_df = user_df[user_df["video_id"].isin(id_to_row.keys())].copy()

    if user_df.empty:
        raise ValueError(f"user_id={user_id} has no videos with embeddings.")

    # 正反馈历史：watch_ratio >= 阈值
    profile_df = user_df[user_df["watch_ratio"] >= pos_threshold].copy()

    # 如果阈值太高导致没有正样本，则退化为取该用户 watch_ratio 最高的视频
    if profile_df.empty:
        print(
            f"[WARN] No positive history with watch_ratio >= {pos_threshold}. "
            f"Fallback to top watch_ratio videos."
        )
        profile_df = user_df.copy()

    profile_df = profile_df.sort_values("watch_ratio", ascending=False)
    profile_df = profile_df.head(max_history).copy()

    # watch_ratio 作为兴趣强度权重，但做截断，避免极端重复播放支配整体兴趣
    weights = profile_df["watch_ratio"].to_numpy(dtype="float32")
    weights = np.clip(weights, a_min=0.0, a_max=weight_cap)

    if weights.sum() <= 0:
        weights = np.ones_like(weights, dtype="float32")

    weights = weights / weights.sum()

    profile_video_ids = profile_df["video_id"].astype(int).to_numpy()
    profile_rows = np.array([id_to_row[int(vid)] for vid in profile_video_ids], dtype=np.int64)

    return user_df, profile_df, profile_rows, weights


# 兴趣向量聚合：对用户正反馈 item 向量做加权平均，得到一个用户兴趣向量
def aggregate_user_interest(embeddings: np.ndarray, profile_rows: np.ndarray, weights: np.ndarray):
    profile_embeddings = embeddings[profile_rows].astype("float32")
    user_vector = np.sum(profile_embeddings * weights[:, None], axis=0, keepdims=True)
    user_vector = user_vector.astype("float32")
    user_vector = np.ascontiguousarray(user_vector)

    if np.isnan(user_vector).any() or np.isinf(user_vector).any():
        raise ValueError("User interest vector contains NaN or Inf.")

    return user_vector


# ※※ 推荐生成：用用户兴趣向量检索 FAISS，并按策略过滤已看或画像 item
def recommend_for_user(
    user_id: int,
    user_vector: np.ndarray,
    index,
    small_user_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    id_to_text: dict,
    metric: str,
    topk: int,
    exclude_mode: str,
    candidate_k: int,
):
    query_vector = user_vector.copy().astype("float32")
    query_vector = np.ascontiguousarray(query_vector)

    if metric == "cosine":
        faiss.normalize_L2(query_vector)

    profile_video_set = set(profile_df["video_id"].astype(int).tolist())
    seen_video_set = set(small_user_df["video_id"].astype(int).tolist())

    if exclude_mode == "profile":
        exclude_set = profile_video_set
    elif exclude_mode == "all_seen":
        exclude_set = seen_video_set
    elif exclude_mode == "none":
        exclude_set = set()
    else:
        raise ValueError(f"Unsupported exclude_mode: {exclude_mode}")

    # KuaiRec small_matrix 很密，all_seen 可能几乎过滤掉全部视频。
    # 因此 all_seen 模式下直接检索全库，避免 candidate_k 不够。
    if exclude_mode == "all_seen":
        search_k = index.ntotal
    else:
        search_k = min(index.ntotal, max(candidate_k, topk + len(exclude_set) + 20))

    # 在 FAISS 索引中，为 query_vector 搜索最相近的 search_k 个向量，得到它们的 相似度分数 和 video_id
    scores, retrieved_ids = index.search(query_vector, search_k) 

    user_watch_ratio = dict(
        zip(
            small_user_df["video_id"].astype(int),
            small_user_df["watch_ratio"].astype(float),
        )
    )

    # 逐个遍历 FAISS 返回的候选视频，按过滤策略生成最终推荐列表
    results = []
    for score, vid in zip(scores[0], retrieved_ids[0]):
        # FAISS 返回的是 numpy 数值类型，转成 Python int 便于集合判断和保存
        vid = int(vid)

        # FAISS 在候选不足时可能返回 -1，表示无效结果，直接跳过
        if vid == -1:
            continue

        # 过滤画像视频或已看视频，具体过滤集合由 exclude_mode 决定
        if vid in exclude_set:
            continue

        # 通过过滤后，将候选视频加入推荐结果
        results.append(
            {
                # rank 按加入结果的顺序生成，也就是过滤后的推荐排序
                "rank": len(results) + 1,
                "user_id": user_id,
                "video_id": vid,
                # score 是 FAISS 返回的相似度分数；cosine 模式下越大越相似
                "score": float(score),
                # 如果这个推荐视频用户曾经看过，则记录历史 watch_ratio；否则为 NaN
                "user_watch_ratio_if_seen": user_watch_ratio.get(vid, np.nan),
                # 补充视频文本，方便人工检查推荐内容是否合理
                "video_text": id_to_text.get(vid, ""),
            }
        )

        # 推荐结果达到 topk 后停止，不再继续遍历候选池
        if len(results) >= topk:
            break

    # 返回推荐结果表，以及本次实际使用的过滤集合
    return pd.DataFrame(results), exclude_set


# 画像展示：打印用于构建用户兴趣向量的高权重历史视频
def print_profile(user_id: int, profile_df: pd.DataFrame, id_to_text: dict, max_print: int = 10):
    print("=" * 100)
    print("[USER INTEREST PROFILE]")
    print(f"user_id: {user_id}")
    print(f"profile videos used: {len(profile_df)}")
    print(f"show top {min(max_print, len(profile_df))} profile videos:")

    for _, row in profile_df.head(max_print).iterrows():
        vid = int(row["video_id"])
        wr = float(row["watch_ratio"])
        print("-" * 100)
        print(f"video_id: {vid}")
        print(f"watch_ratio: {wr:.4f}")
        print("video_text:")
        print(str(id_to_text.get(vid, ""))[:600])


# 推荐展示：打印最终推荐 item 及其分数
def print_recommendations(result_df: pd.DataFrame):
    print("=" * 100)
    print("[RECOMMENDED VIDEOS]")

    if result_df.empty:
        print("No recommendation results.")
        return

    for _, row in result_df.iterrows():
        print("-" * 100)
        print(f"rank: {int(row['rank'])}")
        print(f"video_id: {int(row['video_id'])}")
        print(f"score: {row['score']:.6f}")

        if not pd.isna(row["user_watch_ratio_if_seen"]):
            print(f"user_watch_ratio_if_seen: {row['user_watch_ratio_if_seen']:.4f}")

        print("video_text:")
        print(str(row["video_text"])[:800])


# 结果保存：保存推荐列表和本次用户画像，方便线下检查
def save_results(
    user_id: int,
    topk: int,
    exclude_mode: str,
    result_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    id_to_text: dict,
) -> tuple[Path, Path]:
    out_path = REPORT_DIR / f"user_recommend_{user_id}_top{topk}_{exclude_mode}.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    profile_out_path = REPORT_DIR / f"user_profile_{user_id}.csv"
    profile_save = profile_df.copy()
    profile_save["video_text"] = profile_save["video_id"].astype(int).map(id_to_text)
    profile_save.to_csv(profile_out_path, index=False, encoding="utf-8-sig")
    return out_path, profile_out_path


def main():
    args = parse_args()

    print("=" * 100)
    print("User Interest Recommendation - KuaiRec V1")
    print("=" * 100)
    print(f"[INFO] user_id: {args.user_id}")
    print(f"[INFO] topk: {args.topk}")
    print(f"[INFO] pos_threshold: {args.pos_threshold}")
    print(f"[INFO] max_history: {args.max_history}")
    print(f"[INFO] exclude_mode: {args.exclude_mode}")

    user_history_df, index, embeddings, video_ids, meta_df, index_config = load_resources(args)
    id_to_row, id_to_text = build_lookup(video_ids, meta_df)

    metric = index_config.get("metric", "cosine")
    if metric not in {"cosine", "ip", "l2"}:
        raise ValueError(f"Unsupported index metric: {metric}")

    print("=" * 100)
    print("[DATA]")
    print("user history shape:", user_history_df.shape)
    print("user watched videos:", user_history_df["video_id"].nunique())
    print("num videos in embeddings:", len(video_ids))
    print("index.ntotal:", index.ntotal)
    print("metric:", metric)

    user_df, profile_df, profile_rows, weights = build_user_profile(
        user_id=args.user_id,
        user_df=user_history_df,
        id_to_row=id_to_row,
        max_history=args.max_history,
        pos_threshold=args.pos_threshold,
        weight_cap=args.weight_cap,
    )

    user_vector = aggregate_user_interest(
        embeddings=embeddings,
        profile_rows=profile_rows,
        weights=weights,
    )

    result_df, exclude_set = recommend_for_user(
        user_id=args.user_id,
        user_vector=user_vector,
        index=index,
        small_user_df=user_df,
        profile_df=profile_df,
        id_to_text=id_to_text,
        metric=metric,
        topk=args.topk,
        exclude_mode=args.exclude_mode,
        candidate_k=args.candidate_k,
    )

    print_profile(args.user_id, profile_df, id_to_text, max_print=10)

    print("=" * 100)
    print("[FILTER]")
    print("exclude_mode:", args.exclude_mode)
    print("excluded videos:", len(exclude_set))

    print_recommendations(result_df)

    if args.save:
        out_path, profile_out_path = save_results(
            user_id=args.user_id,
            topk=args.topk,
            exclude_mode=args.exclude_mode,
            result_df=result_df,
            profile_df=profile_df,
            id_to_text=id_to_text,
        )
        print("=" * 100)
        print(f"[SAVED] {out_path}")
        print(f"[SAVED] {profile_out_path}")

    print("=" * 100)
    print("User interest recommendation done.")


if __name__ == "__main__":
    main()
