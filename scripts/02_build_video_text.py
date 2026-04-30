from pathlib import Path
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def find_file(filename: str) -> Path:
    matches = list(RAW_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename} under {RAW_DIR}")
    return matches[0]


def read_csv_kuairec(path: Path) -> pd.DataFrame:
    """
    KuaiRec CSV 中部分文本字段可能包含 \\r。
    显式指定 lineterminator="\\n"，避免 pandas 默认解析时因 \\r 导致行错乱。
    不使用 on_bad_lines='skip'，避免静默丢失样本。
    """
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        lineterminator="\n",
        low_memory=False,
    )


def read_unique_video_ids(path: Path, chunksize: int = 500_000) -> set[int]:
    video_ids: set[int] = set()
    for chunk in pd.read_csv(
        path,
        encoding="utf-8-sig",
        lineterminator="\n",
        usecols=["video_id"],
        dtype={"video_id": "Int32"},
        chunksize=chunksize,
    ):
        video_ids.update(chunk["video_id"].dropna().astype(int).unique())
    return video_ids


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.strip()

    # 去掉过多空白符
    x = re.sub(r"\s+", " ", x)

    # 去掉常见空值字符串
    if x.lower() in {"nan", "none", "null", "unknown", "[]"}:
        return ""

    return x


def build_one_video_text(row: pd.Series, text_cols: list[str]) -> str:
    parts = []

    field_prefix = {
        "manual_cover_text": "封面文字",
        "caption": "视频标题或简介",
        "topic_tag": "话题标签",
        "first_level_category_name": "一级类目",
        "second_level_category_name": "二级类目",
        "third_level_category_name": "三级类目",
    }

    for col in text_cols:
        value = clean_text(row.get(col, ""))
        if value:
            prefix = field_prefix.get(col, col)
            parts.append(f"{prefix}：{value}")

    if not parts:
        return "无文本信息"

    return "。".join(parts) + "。"


def main():
    print("=" * 80)
    print("Build video_text for KuaiRec V1")
    print("=" * 80)

    caption_path = find_file("kuairec_caption_category.csv")
    small_path = find_file("small_matrix.csv")

    print(f"[INFO] caption file: {caption_path}")
    print(f"[INFO] small matrix file: {small_path}")

    caption_df = read_csv_kuairec(caption_path)
    small_video_ids = read_unique_video_ids(small_path)

    print(f"[INFO] caption_df shape: {caption_df.shape}")
    print(f"[INFO] small matrix unique videos: {len(small_video_ids)}")
    print(f"[INFO] caption columns: {list(caption_df.columns)}")

    if "video_id" not in caption_df.columns:
        raise ValueError("kuairec_caption_category.csv must contain video_id column.")

    # V1 只保留 small_matrix 中出现过的视频，避免无关视频进入召回库
    caption_df = caption_df[caption_df["video_id"].isin(small_video_ids)].copy()

    # 根据实际存在的列动态选择文本字段
    candidate_text_cols = [
        "manual_cover_text",
        "caption",
        "topic_tag",
        "first_level_category_name",
        "second_level_category_name",
        "third_level_category_name",
    ]

    text_cols = [col for col in candidate_text_cols if col in caption_df.columns]

    print(f"[INFO] used text columns: {text_cols}")

    if not text_cols:
        raise ValueError(
            "No usable text columns found. Please check kuairec_caption_category.csv columns."
        )

    caption_df["video_text"] = caption_df.apply(
        lambda row: build_one_video_text(row, text_cols),
        axis=1
    )

    out_cols = ["video_id"] + text_cols + ["video_text"]
    video_text_df = caption_df[out_cols].drop_duplicates("video_id").copy()

    out_path = PROCESSED_DIR / "video_text.csv"
    video_text_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[RESULT]")
    print(f"saved to: {out_path}")
    print(f"num videos: {video_text_df['video_id'].nunique()}")
    print(f"shape: {video_text_df.shape}")

    print("=" * 80)
    print("[SAMPLE]")
    sample_df = video_text_df[["video_id", "video_text"]].head(10)
    for _, row in sample_df.iterrows():
        print("-" * 80)
        print("video_id:", row["video_id"])
        print("video_text:", row["video_text"][:500])

    print("=" * 80)
    print("Build video_text done.")


if __name__ == "__main__":
    main()
