from pathlib import Path
import argparse
import json
import math
import random

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EMB_DIR = PROJECT_ROOT / "outputs" / "embeddings"
INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 命令行参数：控制评估口径、用户采样规模，以及行为表/向量/索引文件位置
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V1 user-interest semantic recall.")

    parser.add_argument("--topk", type=int, default=10, help="Evaluate Recall/HitRate/NDCG at top-k.")
    parser.add_argument("--pos-threshold", type=float, default=1.0, help="watch_ratio threshold used to define positive feedback.")
    parser.add_argument("--test-size", type=int, default=1, help="Number of latest positive videos held out per user.")
    parser.add_argument("--max-history", type=int, default=50, help="Maximum positive history videos used to build user profile.")
    parser.add_argument("--weight-cap", type=float, default=5.0, help="Cap watch_ratio weights before user-vector aggregation.")
    parser.add_argument("--candidate-k", type=int, default=500, help="Number of FAISS candidates retrieved before filtering.")
    parser.add_argument("--max-users", type=int, default=0, help="0 means all users.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random baseline and optional user limiting.")

    parser.add_argument("--small-matrix", type=str, default="", help="Path to small_matrix.csv. If empty, search under data/raw.")
    parser.add_argument("--embeddings", type=str, default=str(EMB_DIR / "video_text_embeddings.npy"), help="Path to video text embeddings.")
    parser.add_argument("--video-ids", type=str, default=str(EMB_DIR / "video_ids.npy"), help="Path to video ids.")
    parser.add_argument("--index-path", type=str, default=str(INDEX_DIR / "video_text_faiss.index"), help="Path to FAISS index.")
    parser.add_argument("--index-config", type=str, default=str(INDEX_DIR / "faiss_index_config.json"), help="Path to FAISS index config.")

    args = parser.parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive.")
    if args.test_size <= 0:
        raise ValueError("--test-size must be positive.")
    if args.max_history <= 0:
        raise ValueError("--max-history must be positive.")
    if args.weight_cap <= 0:
        raise ValueError("--weight-cap must be positive.")
    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be positive.")
    if args.max_users < 0:
        raise ValueError("--max-users cannot be negative.")
    return args


# 数据定位：在 data/raw 下递归查找 KuaiRec 原始文件
def find_file(filename: str) -> Path:
    matches = list(RAW_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename} under {RAW_DIR}")
    return matches[0]


# CSV 读取：只读评估所需列，并保留 KuaiRec 的 \n 行结束兼容处理
def read_csv_kuairec(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        lineterminator="\n",
        low_memory=False,
        usecols=usecols,
    )


# 行为表字段选择：优先使用 timestamp 做时间切分，缺失时退回 date
def get_behavior_usecols(path: Path) -> list[str]:
    header = pd.read_csv(path, encoding="utf-8-sig", lineterminator="\n", nrows=0)
    usecols = ["user_id", "video_id", "watch_ratio"]
    if "timestamp" in header.columns:
        usecols.append("timestamp")
    elif "date" in header.columns:
        usecols.append("date")
    return usecols


# JSON 读取：加载 FAISS 建索引阶段保存的配置
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 资源加载：读取行为表、item 向量、video_id 映射、FAISS 索引和索引配置
def load_resources(args):
    small_path = Path(args.small_matrix) if args.small_matrix else find_file("small_matrix.csv")
    emb_path = Path(args.embeddings)
    ids_path = Path(args.video_ids)
    index_path = Path(args.index_path)
    index_config_path = Path(args.index_config)

    if not small_path.exists():
        raise FileNotFoundError(f"small_matrix not found: {small_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Video ids not found: {ids_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    print(f"[INFO] reading small_matrix: {small_path}")
    small_df = read_csv_kuairec(
        small_path,
        usecols=get_behavior_usecols(small_path),
    )

    embeddings = np.load(emb_path).astype("float32")
    video_ids = np.load(ids_path).astype("int64")
    embeddings = np.ascontiguousarray(embeddings)
    video_ids = np.ascontiguousarray(video_ids)
    index = faiss.read_index(str(index_path))
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

    id_to_row = {int(vid): i for i, vid in enumerate(video_ids.tolist())}
    video_id_set = set(id_to_row.keys())

    small_df["user_id"] = pd.to_numeric(small_df["user_id"], errors="raise").astype("int64")
    small_df["video_id"] = pd.to_numeric(small_df["video_id"], errors="raise").astype("int64")
    small_df["watch_ratio"] = pd.to_numeric(small_df["watch_ratio"], errors="coerce").fillna(0.0).astype("float32")
    if "timestamp" in small_df.columns:
        small_df["timestamp"] = pd.to_numeric(small_df["timestamp"], errors="coerce")
    if "date" in small_df.columns:
        small_df["date"] = pd.to_numeric(small_df["date"], errors="coerce")

    # 当前 V1 embedding 覆盖 small_matrix 全部视频；这里保留过滤，防止以后换底库时出现缺失
    small_df = small_df[small_df["video_id"].isin(video_id_set)].copy()

    return small_df, embeddings, video_ids, index, index_config, id_to_row


# 单用户切分：按时间留出最后 test_size 个正反馈 item 作为测试目标
def split_user_history(user_df: pd.DataFrame, pos_threshold: float, test_size: int):
    pos_df = user_df[user_df["watch_ratio"] >= pos_threshold].copy()

    if len(pos_df) <= test_size:
        return None, None

    if "timestamp" in pos_df.columns:
        pos_df = pos_df.sort_values("timestamp")
    elif "date" in pos_df.columns:
        pos_df = pos_df.sort_values("date")
    else:
        pos_df = pos_df.sort_values("watch_ratio")

    test_df = pos_df.tail(test_size).copy()
    train_df = pos_df.iloc[:-test_size].copy()

    if train_df.empty or test_df.empty:
        return None, None

    return train_df, test_df


# 用户向量构建：用训练正反馈 item 的文本向量，按 watch_ratio 加权平均
def build_user_vector(
    train_df: pd.DataFrame,
    embeddings: np.ndarray,
    id_to_row: dict,
    max_history: int,
    weight_cap: float,
):
    profile_df = train_df.sort_values("watch_ratio", ascending=False).head(max_history).copy()

    profile_video_ids = profile_df["video_id"].astype(int).to_numpy()
    rows = np.array([id_to_row[int(vid)] for vid in profile_video_ids], dtype=np.int64)

    weights = profile_df["watch_ratio"].to_numpy(dtype="float32")
    weights = np.clip(weights, a_min=0.0, a_max=weight_cap)

    if weights.sum() <= 0:
        weights = np.ones_like(weights, dtype="float32")

    weights = weights / weights.sum()

    user_vec = np.sum(embeddings[rows] * weights[:, None], axis=0, keepdims=True)
    user_vec = user_vec.astype("float32")
    user_vec = np.ascontiguousarray(user_vec)
    if np.isnan(user_vec).any() or np.isinf(user_vec).any():
        raise ValueError("User vector contains NaN or Inf.")

    return user_vec, set(profile_video_ids.tolist())


# 语义召回：使用用户向量搜索 FAISS，过滤训练画像 item，返回 topk 推荐 video_id
def semantic_recommend(
    user_vec: np.ndarray,
    index,
    metric: str,
    exclude_set: set[int],
    topk: int,
    candidate_k: int,
):
    query = user_vec.copy().astype("float32")
    query = np.ascontiguousarray(query)

    if metric == "cosine":
        faiss.normalize_L2(query)

    search_k = min(index.ntotal, max(candidate_k, topk + len(exclude_set) + 20))
    scores, ids = index.search(query, search_k)

    recs = []
    for vid in ids[0]:
        vid = int(vid)
        if vid == -1:
            continue
        if vid in exclude_set:
            continue
        recs.append(vid)
        if len(recs) >= topk:
            break

    return recs


# 热门基线：按全局正反馈热度排序，过滤训练画像 item 后返回 topk
def popularity_recommend(popular_video_ids: list[int], exclude_set: set[int], topk: int):
    recs = []
    for vid in popular_video_ids:
        if vid in exclude_set:
            continue
        recs.append(int(vid))
        if len(recs) >= topk:
            break
    return recs


# 随机基线：在未过滤 item 中随机采样 topk，作为最低参考线
def random_recommend(all_video_ids: list[int], exclude_set: set[int], topk: int, rng: random.Random):
    candidates = [int(v) for v in all_video_ids if int(v) not in exclude_set]

    if len(candidates) <= topk:
        return candidates

    return rng.sample(candidates, topk)


# 指标计算：计算单个用户在 topk 下的 HitRate、Recall 和 NDCG
def calc_metrics(recs: list[int], targets: set[int], topk: int):
    recs = recs[:topk]
    hits = [1 if vid in targets else 0 for vid in recs]
    num_hits = sum(hits)

    hitrate = 1.0 if num_hits > 0 else 0.0
    recall = num_hits / max(len(targets), 1)

    dcg = 0.0
    for i, hit in enumerate(hits):
        if hit:
            rank = i + 1
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(targets), topk)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "hitrate": hitrate,
        "recall": recall,
        "ndcg": ndcg,
        "num_hits": num_hits,
    }


# 汇总指标：对所有有效用户取均值，得到每种方法的整体表现
def summarize(result_df: pd.DataFrame, topk: int):
    rows = []

    for method, group in result_df.groupby("method"):
        rows.append(
            {
                "method": method,
                f"HitRate@{topk}": group["hitrate"].mean(),
                f"Recall@{topk}": group["recall"].mean(),
                f"NDCG@{topk}": group["ndcg"].mean(),
                "num_users": group["user_id"].nunique(),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(f"Recall@{topk}", ascending=False).reset_index(drop=True)

    return summary_df


# 结果保存：保存逐用户明细指标和方法级汇总指标
def save_results(result_df: pd.DataFrame, summary_df: pd.DataFrame, topk: int, pos_threshold: float) -> tuple[Path, Path]:
    out_detail = REPORT_DIR / f"eval_detail_top{topk}_thr{pos_threshold}.csv"
    out_summary = REPORT_DIR / f"eval_summary_top{topk}_thr{pos_threshold}.csv"

    result_df.to_csv(out_detail, index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_summary, index=False, encoding="utf-8-sig")
    return out_detail, out_summary


def main():
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    print("=" * 100)
    print("Evaluate KuaiRec V1 Recall Metrics")
    print("=" * 100)
    print(f"[INFO] topk: {args.topk}")
    print(f"[INFO] pos_threshold: {args.pos_threshold}")
    print(f"[INFO] test_size: {args.test_size}")
    print(f"[INFO] max_history: {args.max_history}")
    print(f"[INFO] max_users: {args.max_users if args.max_users > 0 else 'all'}")

    small_df, embeddings, video_ids, index, index_config, id_to_row = load_resources(args)
    metric = index_config.get("metric", "cosine")
    if metric not in {"cosine", "ip", "l2"}:
        raise ValueError(f"Unsupported index metric: {metric}")

    print("=" * 100)
    print("[DATA]")
    print("small_matrix shape:", small_df.shape)
    print("num users:", small_df["user_id"].nunique())
    print("num videos:", len(video_ids))
    print("embedding shape:", embeddings.shape)
    print("index.ntotal:", index.ntotal)
    print("metric:", metric)

    # ========== 1. 构造热门推荐 baseline ==========
    # 只使用正反馈行为统计热门视频，避免低观看完成度的视频进入热门榜
    pop_df = (
        # 先筛选正反馈记录：默认 watch_ratio >= 1.0
        small_df[small_df["watch_ratio"] >= args.pos_threshold]
        # 按视频聚合，统计每个视频在全局范围内的受欢迎程度
        .groupby("video_id")
        # pos_count 表示正反馈次数，mean_watch_ratio 表示正反馈记录的平均观看比例
        .agg(
            pos_count=("watch_ratio", "size"),
            mean_watch_ratio=("watch_ratio", "mean"),
        )
        # 将 groupby 后作为索引的 video_id 恢复成普通列
        .reset_index()
        # 先按正反馈次数降序，再按平均观看比例降序，得到全局热门排序
        .sort_values(["pos_count", "mean_watch_ratio"], ascending=False)
    )

    # 取出排序后的 video_id，后续 popularity_recommend 会从这个列表头部开始推荐
    popular_video_ids = pop_df["video_id"].astype(int).tolist()

    # ========== 2. 准备随机推荐候选池 ==========
    # 将全部有 embedding 的视频 ID 转成 Python int 列表，供 random baseline 随机采样
    all_video_ids = [int(v) for v in video_ids.tolist()]

    # ========== 3. 按用户分组并控制评估用户数 ==========
    # 将行为表按 user_id 分组，每一组就是一个用户的全部观看记录
    user_groups = list(small_df.groupby("user_id", sort=False))

    # max_users > 0 时只评估前 max_users 个用户，便于快速调试
    if args.max_users > 0:
        user_groups = user_groups[: args.max_users]

    # ========== 4. 初始化评估结果容器 ==========
    # records 用于保存每个用户、每种推荐方法的评估明细
    records = []

    # skipped_users 统计因正反馈不足等原因被跳过的用户数量
    skipped_users = 0

    # ========== 5. 逐用户生成推荐并计算指标 ==========
    # 遍历每个用户的行为记录，并用 tqdm 显示评估进度
    for user_id, user_df in tqdm(user_groups, desc="Evaluating users"):
        # 将 user_id 转成普通 int，避免保存 CSV 时出现 numpy 标量类型
        user_id = int(user_id)

        # 复制当前用户行为，后续切分/排序不会影响原始 small_df
        user_df = user_df.copy()

        # 将当前用户的正反馈历史切成训练历史和测试目标
        train_df, test_df = split_user_history(
            user_df=user_df,
            pos_threshold=args.pos_threshold,
            test_size=args.test_size,
        )

        # 如果该用户正反馈数量不足，无法同时构造训练集和测试集，则跳过
        if train_df is None or test_df is None:
            skipped_users += 1
            continue

        # 基于训练历史中的视频 embedding，加权平均得到用户兴趣向量
        user_vec, profile_video_set = build_user_vector(
            train_df=train_df,
            embeddings=embeddings,
            id_to_row=id_to_row,
            max_history=args.max_history,
            weight_cap=args.weight_cap,
        )

        # 测试集视频就是当前用户需要被推荐命中的标准答案
        target_set = set(test_df["video_id"].astype(int).tolist())

        # 方法一：语义召回，用用户兴趣向量搜索 FAISS，再过滤画像视频
        semantic_recs = semantic_recommend(
            user_vec=user_vec,
            index=index,
            metric=metric,
            exclude_set=profile_video_set,
            topk=args.topk,
            candidate_k=args.candidate_k,
        )

        # 方法二：热门推荐，从全局热门榜中取结果，并过滤画像视频
        popularity_recs = popularity_recommend(
            popular_video_ids=popular_video_ids,
            exclude_set=profile_video_set,
            topk=args.topk,
        )

        # 方法三：随机推荐，从全库视频中随机采样，并过滤画像视频
        random_recs = random_recommend(
            all_video_ids=all_video_ids,
            exclude_set=profile_video_set,
            topk=args.topk,
            rng=rng,
        )

        # 将三种推荐结果放到同一个字典中，方便统一计算指标
        method_to_recs = {
            "semantic": semantic_recs,
            "popularity": popularity_recs,
            "random": random_recs,
        }

        # 逐个方法计算当前用户的 HitRate/Recall/NDCG
        for method, recs in method_to_recs.items():
            # 对当前方法的推荐列表和测试答案计算 topk 指标
            metrics = calc_metrics(recs, target_set, args.topk)

            # 保存当前用户、当前方法的一条评估明细
            records.append(
                {
                    # 当前用户 ID
                    "user_id": user_id,
                    # 当前推荐方法名称：semantic / popularity / random
                    "method": method,
                    # 用于构建用户画像的训练正反馈数量
                    "num_train_pos": len(train_df),
                    # 留作测试答案的正反馈数量
                    "num_test_pos": len(test_df),
                    # 当前用户的测试目标视频 ID，多个目标用逗号拼接
                    "target_video_ids": ",".join(map(str, sorted(target_set))),
                    # 当前方法给出的推荐视频 ID 列表，按推荐顺序用逗号拼接
                    "recommended_video_ids": ",".join(map(str, recs)),
                    # 展开 hitrate、recall、ndcg、num_hits 等指标字段
                    **metrics,
                }
            )

    # ========== 6. 汇总逐用户评估明细 ==========
    # 将 records 列表转成 DataFrame，每行对应一个用户在一种方法下的评估结果
    result_df = pd.DataFrame(records)

    # 如果没有任何有效用户，说明评估口径过严，需要降低 pos_threshold 或 test_size
    if result_df.empty:
        raise RuntimeError("No valid users for evaluation. Try lowering --pos-threshold.")

    # 按推荐方法汇总平均 HitRate@K、Recall@K、NDCG@K
    summary_df = summarize(result_df, args.topk)

    # 保存逐用户明细表和方法级汇总表
    out_detail, out_summary = save_results(result_df, summary_df, args.topk, args.pos_threshold)

    # ========== 7. 打印汇总指标 ==========
    # 输出分隔线，方便在终端中阅读结果
    print("=" * 100)

    # 打印汇总结果标题
    print("[SUMMARY]")

    # 打印每种方法的整体指标表现
    print(summary_df)

    # ========== 8. 打印输出文件路径 ==========
    # 输出分隔线
    print("=" * 100)

    # 打印保存路径标题
    print("[SAVED]")

    # 打印逐用户明细结果路径
    print(f"detail: {out_detail}")

    # 打印方法级汇总结果路径
    print(f"summary: {out_summary}")

    # ========== 9. 打印评估完成信息 ==========
    # 输出分隔线
    print("=" * 100)

    # 打印评估统计标题
    print("[INFO]")

    # 打印实际参与评估的用户数
    print("evaluated users:", result_df["user_id"].nunique())

    # 打印因数据不足被跳过的用户数
    print("skipped users:", skipped_users)

    # 打印完成提示
    print("Evaluation done.")


if __name__ == "__main__":
    main()
