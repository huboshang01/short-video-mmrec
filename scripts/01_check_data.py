from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def find_file(filename: str) -> Path:
    matches = list(RAW_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename} under {RAW_DIR}")
    return matches[0]


def print_basic_info(
    name: str,
    path: Path,
    nrows: int | None = None,
    **read_csv_kwargs,
):
    print("=" * 80)
    print(f"[{name}]")
    print(f"path: {path}")

    df = pd.read_csv(path, nrows=nrows, **read_csv_kwargs)
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("head:")
    print(df.head())

    return df


def main():
    print("=" * 80)
    print("KuaiRec Data Check")
    print("=" * 80)
    print(f"project root: {PROJECT_ROOT}")
    print(f"raw dir: {RAW_DIR}")

    required_files = [
        "small_matrix.csv",
        "item_categories.csv",
        "item_daily_features.csv",
        "user_features.csv",
        "kuairec_caption_category.csv",
    ]

    paths = {}
    for filename in required_files:
        path = find_file(filename)
        paths[filename] = path
        print(f"[FOUND] {filename}: {path}")

    # 1. small_matrix 是 V1 主交互表，完整读取
    small = pd.read_csv(paths["small_matrix.csv"])
    print("=" * 80)
    print("[small_matrix.csv]")
    print(f"shape: {small.shape}")
    print(f"columns: {list(small.columns)}")
    print(small.head())

    print("-" * 80)
    print("small_matrix key stats:")
    print("num users:", small["user_id"].nunique())
    print("num videos:", small["video_id"].nunique())
    print("num interactions:", len(small))
    print("watch_ratio describe:")
    print(small["watch_ratio"].describe())

    # 2. 视频类目
    item_categories = print_basic_info(
        "item_categories.csv",
        paths["item_categories.csv"]
    )

    # 3. 视频每日统计，只读前几行看字段
    item_daily = print_basic_info(
        "item_daily_features.csv",
        paths["item_daily_features.csv"],
        nrows=5
    )

    # 4. 用户特征
    user_features = print_basic_info(
        "user_features.csv",
        paths["user_features.csv"]
    )

    # 5. 文本字段，这是 V1 语义召回最关键文件
    caption = print_basic_info(
        "kuairec_caption_category.csv",
        paths["kuairec_caption_category.csv"],
        lineterminator="\n",
    )

    caption_cr_count = int(
        caption.select_dtypes(include="object")
        .apply(lambda col: col.str.contains("\r", regex=False, na=False).sum())
        .sum()
    )
    print(f"caption fields containing raw carriage returns: {caption_cr_count}")

    print("=" * 80)
    print("Text file columns candidate:")
    for col in caption.columns:
        print("-", col)

    print("=" * 80)
    print("Data check done.")


if __name__ == "__main__":
    main()
