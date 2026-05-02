from pathlib import Path
import argparse
import ast
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMALL_MATRIX = PROJECT_ROOT / "data" / "raw" / "KuaiRec 2.0" / "data" / "small_matrix.csv"
DEFAULT_VIDEO_TEXT = PROJECT_ROOT / "data" / "processed" / "video_text.csv"
DEFAULT_ITEM_CATEGORIES = PROJECT_ROOT / "data" / "raw" / "KuaiRec 2.0" / "data" / "item_categories.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "v2"


# 命令行参数：控制输入文件、正反馈阈值、用户过滤阈值和切分比例
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare V2 behavior samples from KuaiRec interactions.")

    parser.add_argument("--small-matrix", type=str, default=str(DEFAULT_SMALL_MATRIX), help="Path to small_matrix.csv.")
    parser.add_argument("--video-text", type=str, default=str(DEFAULT_VIDEO_TEXT), help="Path to V1 video_text.csv.")
    parser.add_argument("--item-categories", type=str, default=str(DEFAULT_ITEM_CATEGORIES), help="Path to item_categories.csv.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for V2 behavior samples.")
    parser.add_argument("--watch-threshold", type=float, default=1.0, help="watch_ratio threshold used to mark positive feedback.")
    parser.add_argument("--min-user-interactions", type=int, default=5, help="Drop users with fewer behavior samples than this value.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Per-user validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Per-user test split ratio.")

    args = parser.parse_args()
    if args.min_user_interactions < 2:
        raise ValueError("--min-user-interactions must be at least 2.")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1).")
    if not 0 <= args.test_ratio < 1:
        raise ValueError("--test-ratio must be in [0, 1).")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio + --test-ratio must be less than 1.")
    return args


# 路径处理：支持传入绝对路径，也支持相对于项目根目录的相对路径
def resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


# CSV 读取：统一 KuaiRec CSV 的编码和行结束符处理
def read_csv_kuairec(path: Path, usecols=None, nrows=None) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", lineterminator="\n", low_memory=False, usecols=usecols, nrows=nrows)


# 字段匹配：兼容不同数据文件里的同义列名
def find_col(columns, candidates, required=True, name="column"):
    cols = list(columns)
    lower_map = {col.lower(): col for col in cols}

    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if required:
        raise ValueError(f"Cannot find {name}. Candidates={candidates}, existing={cols}")
    return None


# 行为表读取：只读取训练样本构造需要的列，避免一次性加载无关字段
def load_interactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"small_matrix not found: {path}")

    header = read_csv_kuairec(path, nrows=0)
    user_col = find_col(header.columns, ["user_id", "uid", "user"], name="user_id")
    item_col = find_col(header.columns, ["video_id", "item_id", "itemid", "video"], name="item_id")
    watch_col = find_col(header.columns, ["watch_ratio", "play_ratio", "watch_ratio_norm"], required=False, name="watch_ratio")
    time_cols = []
    for time_name in ["timestamp", "time", "date"]:
        time_col = find_col(header.columns, [time_name], required=False, name=time_name)
        if time_col is not None and time_col not in time_cols:
            time_cols.append(time_col)

    usecols = [user_col, item_col]
    if watch_col is not None:
        usecols.append(watch_col)
    else:
        play_col = find_col(header.columns, ["play_duration", "play_time"], name="play_duration")
        duration_col = find_col(header.columns, ["video_duration", "duration"], name="video_duration")
        usecols.extend([play_col, duration_col])

    usecols.extend(time_cols)

    interactions = read_csv_kuairec(path, usecols=usecols)
    interactions = interactions.rename(columns={user_col: "user_id", item_col: "item_id"})

    if watch_col is not None:
        interactions["watch_ratio"] = pd.to_numeric(interactions[watch_col], errors="coerce")
    else:
        play = pd.to_numeric(interactions[play_col], errors="coerce")
        duration = pd.to_numeric(interactions[duration_col], errors="coerce").replace(0, np.nan)
        interactions["watch_ratio"] = play / duration

    if time_cols:
        sort_key = pd.to_numeric(interactions[time_cols[0]], errors="coerce")
        for time_col in time_cols[1:]:
            sort_key = sort_key.fillna(pd.to_numeric(interactions[time_col], errors="coerce"))
        interactions["_sort_key"] = sort_key.fillna(pd.Series(np.arange(len(interactions), dtype=np.int64), index=interactions.index))
    else:
        interactions["_sort_key"] = np.arange(len(interactions), dtype=np.int64)

    interactions["user_id"] = pd.to_numeric(interactions["user_id"], errors="coerce")
    interactions["item_id"] = pd.to_numeric(interactions["item_id"], errors="coerce")
    interactions = interactions.dropna(subset=["user_id", "item_id", "watch_ratio"]).copy()
    interactions["user_id"] = interactions["user_id"].astype("int64")
    interactions["item_id"] = interactions["item_id"].astype("int64")

    return interactions[["user_id", "item_id", "watch_ratio", "_sort_key"]]


# 文本表构造：优先使用 V1 生成的 video_text；缺失时拼接其它文本列兜底
def build_text_column(video_text_df: pd.DataFrame, item_col: str) -> pd.DataFrame:
    text_col = find_col(
        video_text_df.columns,
        ["item_text", "video_text", "text", "caption", "title", "description", "combined_text"],
        required=False,
        name="item text",
    )

    if text_col is not None:
        out = video_text_df[[item_col, text_col]].copy()
        out = out.rename(columns={item_col: "item_id", text_col: "item_text"})
    else:
        text_cols = [col for col in video_text_df.columns if col != item_col and video_text_df[col].dtype == "object"]
        if not text_cols:
            raise ValueError(f"Cannot build item_text from columns: {list(video_text_df.columns)}")
        out = video_text_df[[item_col] + text_cols].copy()
        out["item_text"] = out[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        out = out[[item_col, "item_text"]].rename(columns={item_col: "item_id"})

    out["item_id"] = pd.to_numeric(out["item_id"], errors="raise").astype("int64")
    out["item_text"] = out["item_text"].fillna("").astype(str).str.strip()
    out.loc[out["item_text"] == "", "item_text"] = "无文本信息"
    return out.drop_duplicates("item_id").reset_index(drop=True)


# 类目表构造：将 item_categories.csv 中的 feat 等字段统一命名为 category_id
def extract_primary_category(value) -> str:
    try:
        categories = ast.literal_eval(str(value))
        if isinstance(categories, list) and categories:
            return str(categories[0])
    except (ValueError, SyntaxError):
        pass
    return str(value)


def build_category_column(category_df: pd.DataFrame, item_col: str) -> pd.DataFrame | None:
    cat_col = find_col(
        category_df.columns,
        ["category_id", "category", "category_name", "first_level_category_id", "second_level_category_id", "feat"],
        required=False,
        name="category",
    )

    if cat_col is None:
        non_id_cols = [col for col in category_df.columns if col != item_col]
        if not non_id_cols:
            return None
        cat_col = non_id_cols[0]

    out = category_df[[item_col, cat_col]].copy()
    out = out.rename(columns={item_col: "item_id", cat_col: "category_id"})
    out["item_id"] = pd.to_numeric(out["item_id"], errors="raise").astype("int64")
    out["category_id"] = out["category_id"].fillna("unknown").map(extract_primary_category)
    return out.drop_duplicates("item_id").reset_index(drop=True)


# 用户级时间切分：每个用户内部按行为时间排序，再切 train/val/test
def split_by_user(df: pd.DataFrame, val_ratio: float, test_ratio: float, min_user_interactions: int) -> pd.DataFrame:
    user_cnt = df.groupby("user_id")["item_id"].count()
    keep_users = user_cnt[user_cnt >= min_user_interactions].index
    df = df[df["user_id"].isin(keep_users)].copy()
    df["split"] = "train"

    for _, group in df.groupby("user_id", sort=False):
        group = group.sort_values("_sort_key")
        n = len(group)

        n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
        n_test = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0

        while n - n_val - n_test < 1:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break

        idx = group.index.to_numpy()
        if n_val > 0:
            df.loc[idx[n - n_val - n_test: n - n_test], "split"] = "val"
        if n_test > 0:
            df.loc[idx[n - n_test:], "split"] = "test"

    return df


# 结果保存：输出全量样本、三份切分样本、item 文本表和统计摘要
def save_outputs(samples: pd.DataFrame, out_dir: Path, args) -> None:
    all_path = out_dir / "behavior_samples_all.csv"
    train_path = out_dir / "behavior_samples_train.csv"
    val_path = out_dir / "behavior_samples_val.csv"
    test_path = out_dir / "behavior_samples_test.csv"
    item_text_path = out_dir / "item_text_v2.csv"
    summary_path = out_dir / "behavior_samples_summary.json"

    item_text_v2 = samples[["item_id", "item_text", "category_id"]].drop_duplicates("item_id")

    samples.to_csv(all_path, index=False, encoding="utf-8-sig")
    samples[samples["split"] == "train"].to_csv(train_path, index=False, encoding="utf-8-sig")
    samples[samples["split"] == "val"].to_csv(val_path, index=False, encoding="utf-8-sig")
    samples[samples["split"] == "test"].to_csv(test_path, index=False, encoding="utf-8-sig")
    item_text_v2.to_csv(item_text_path, index=False, encoding="utf-8-sig")

    summary = {
        "num_samples": int(len(samples)),
        "num_users": int(samples["user_id"].nunique()),
        "num_items": int(samples["item_id"].nunique()),
        "num_positive": int(samples["is_positive"].sum()),
        "positive_ratio": float(samples["is_positive"].mean()),
        "watch_ratio_mean": float(samples["watch_ratio"].mean()),
        "watch_ratio_median": float(samples["watch_ratio"].median()),
        "split_counts": {str(k): int(v) for k, v in samples["split"].value_counts().to_dict().items()},
        "watch_threshold": float(args.watch_threshold),
        "min_user_interactions": int(args.min_user_interactions),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Done] V2 behavior samples generated.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[Saved] {all_path}")
    print(f"[Saved] {train_path}")
    print(f"[Saved] {val_path}")
    print(f"[Saved] {test_path}")
    print(f"[Saved] {item_text_path}")
    print(f"[Saved] {summary_path}")


def main():
    args = parse_args()

    small_path = resolve_path(args.small_matrix)
    text_path = resolve_path(args.video_text)
    category_path = resolve_path(args.item_categories)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Prepare V2 Behavior Samples")
    print("=" * 80)
    print(f"[INFO] project root: {PROJECT_ROOT}")
    print(f"[INFO] small_matrix: {small_path}")
    print(f"[INFO] video_text: {text_path}")
    print(f"[INFO] item_categories: {category_path}")
    print(f"[INFO] output dir: {out_dir}")

    print("\n[Load] interactions")
    interactions = load_interactions(small_path)
    print("interactions shape:", interactions.shape)

    print("\n[Load] item text")
    if not text_path.exists():
        raise FileNotFoundError(f"video_text not found: {text_path}")
    video_text = read_csv_kuairec(text_path)
    text_item_col = find_col(video_text.columns, ["video_id", "item_id", "itemid", "video"], name="text item_id")
    item_text = build_text_column(video_text, text_item_col)
    print("item_text shape:", item_text.shape)

    print("\n[Merge] interactions + item text")
    samples = interactions.merge(item_text, on="item_id", how="inner")
    print("samples after text merge:", samples.shape)

    print("\n[Load] item categories")
    if category_path.exists():
        category_df = read_csv_kuairec(category_path)
        cat_item_col = find_col(category_df.columns, ["video_id", "item_id", "itemid", "video"], name="category item_id")
        item_category = build_category_column(category_df, cat_item_col)
        if item_category is not None:
            samples = samples.merge(item_category, on="item_id", how="left")
        else:
            samples["category_id"] = "unknown"
    else:
        print(f"[WARN] item_categories not found, use category_id=unknown: {category_path}")
        samples["category_id"] = "unknown"

    print("\n[Label] positive feedback")
    samples["category_id"] = samples["category_id"].fillna("unknown").astype(str)
    samples["is_positive"] = (samples["watch_ratio"] >= args.watch_threshold).astype("int8")

    print("\n[Split] train / val / test by user time order")
    samples = split_by_user(samples, args.val_ratio, args.test_ratio, args.min_user_interactions)

    samples = samples.sort_values(["user_id", "_sort_key", "item_id"]).reset_index(drop=True)
    samples = samples[["user_id", "item_id", "item_text", "watch_ratio", "category_id", "is_positive", "split"]].copy()

    save_outputs(samples, out_dir, args)


if __name__ == "__main__":
    main()
