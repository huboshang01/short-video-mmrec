"""
V3 Step 01: 构建 MicroLens-100K 召回训练样本。024

本脚本把 MicroLens-100K 原始文件整理成 V3 后续模型训练需要的标准输入：
    1. 读取 user-item-time 隐式交互，并按用户时间序列切分 train / val / test。
    2. 读取 item title 与 likes/views，生成 item_meta.csv。
    3. 生成 item_ids.csv，固定 item_id 到特征矩阵行号的映射。
    4. 生成 feature_config.json，记录 text/image/video 三类官方特征路径和维度。

注意：
    MicroLens-100K 没有 KuaiRec 的 watch_ratio / 显式曝光负反馈。
    因此本阶段只输出 label=1 的正反馈样本；负样本将在 V3 训练 Dataset 中按全量 item
    进行 sampled negative 构造。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import ast
import csv
import json
import struct


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "microlens_100k"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "v3" / "microlens_100k"

PAIRS_NAME = "MicroLens-100k_pairs.csv"
TITLE_NAME = "MicroLens-100k_title_en.csv"
LIKES_VIEWS_NAME = "MicroLens-100k_likes_and_views.txt"
FEATURE_DIR_NAME = "extracted_modality_features"

FEATURE_SPECS = {
    "text": ("MicroLens-100k_title_en_text_features_BgeM3", 1024),
    "image": ("MicroLens-100k_image_features_CLIPRN50", 1024),
    "video": ("MicroLens-100k_video_features_VideoMAE", 768),
}

NO_TITLE = "No title"
UNKNOWN_CATEGORY = "__UNK__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare V3 MicroLens-100K retrieval samples.")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Path to MicroLens-100K raw directory.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Directory for processed V3 MicroLens files.")
    parser.add_argument("--min-user-interactions", type=int, default=3, help="Drop users with fewer interactions than this value.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Per-user validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Per-user test split ratio.")
    parser.add_argument("--source-name", type=str, default="microlens_100k", help="Source tag saved into output samples.")
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


def resolve_project_path(path: str | Path) -> Path:
    """支持命令行传入项目内相对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    """配置文件保存相对项目根目录的路径，便于迁移仓库。"""
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_titles(path: Path) -> dict[int, str]:
    """读取无表头标题文件，并为空标题补兜底文本。"""
    titles: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            item_id = int(row[0].strip())
            title = ",".join(row[1:]).strip() or NO_TITLE
            titles[item_id] = title
    if not titles:
        raise ValueError(f"No titles loaded from {path}")
    return titles


def load_likes_views(path: Path) -> dict[int, tuple[int, int]]:
    """读取 item 热度表：item_id, likes, views。"""
    stats: dict[int, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no} expected 3 tab-separated fields, got {len(parts)}")
            item_id, likes, views = int(parts[0]), int(parts[1]), int(parts[2])
            stats[item_id] = (likes, views)
    if not stats:
        raise ValueError(f"No likes/views rows loaded from {path}")
    return stats


def load_interactions(path: Path, valid_item_ids: set[int]) -> list[dict[str, int]]:
    """读取隐式交互，只保留有 item 内容/特征的记录。"""
    interactions: list[dict[str, int]] = []
    seen_exact_rows: set[tuple[int, int, int]] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"user", "item", "timestamp"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        for row in reader:
            user_id = int(row["user"])
            item_id = int(row["item"])
            timestamp = int(row["timestamp"])
            exact_key = (user_id, item_id, timestamp)

            # 精确重复不会提供新的序列信息，直接跳过可减少训练噪声。
            if exact_key in seen_exact_rows or item_id not in valid_item_ids:
                continue

            seen_exact_rows.add(exact_key)
            interactions.append({"user_id": user_id, "item_id": item_id, "timestamp": timestamp})

    if not interactions:
        raise ValueError(f"No valid interactions loaded from {path}")
    return interactions


def split_by_user(
    interactions: list[dict[str, int]],
    min_user_interactions: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """每个用户内部按时间排序，再切分，保证 val/test 发生在 train 之后。"""
    by_user: dict[int, list[dict[str, int]]] = defaultdict(list)
    for row in interactions:
        by_user[int(row["user_id"])].append(row)

    samples: list[dict[str, object]] = []
    dropped_users = 0
    dropped_interactions = 0

    for user_id, rows in by_user.items():
        rows.sort(key=lambda row: (row["timestamp"], row["item_id"]))
        if len(rows) < min_user_interactions:
            dropped_users += 1
            dropped_interactions += len(rows)
            continue

        n_val = max(1, int(round(len(rows) * val_ratio))) if val_ratio > 0 else 0
        n_test = max(1, int(round(len(rows) * test_ratio))) if test_ratio > 0 else 0

        # 至少保留一条 train 历史，后续 user tower 才能构造用户表示。
        while len(rows) - n_val - n_test < 1:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break

        val_start = len(rows) - n_val - n_test
        test_start = len(rows) - n_test

        for idx, row in enumerate(rows):
            split = "train"
            if n_val > 0 and val_start <= idx < test_start:
                split = "val"
            elif n_test > 0 and idx >= test_start:
                split = "test"

            samples.append(
                {
                    "user_id": user_id,
                    "item_id": int(row["item_id"]),
                    "timestamp": int(row["timestamp"]),
                    "sort_key": int(row["timestamp"]),
                    "label": 1,
                    "sample_weight": 1.0,
                    "category_id": UNKNOWN_CATEGORY,
                    "split": split,
                }
            )

    samples.sort(key=lambda row: (int(row["user_id"]), int(row["sort_key"]), int(row["item_id"])))
    drop_stats = {"dropped_users": dropped_users, "dropped_interactions": dropped_interactions}
    return samples, drop_stats


def parse_npy_shape(path: Path) -> tuple[int, ...]:
    """解析 .npy shape，避免在样本准备阶段加载大矩阵。"""
    with path.open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path} is not a valid .npy file.")
        major, _minor = f.read(2)
        header_len_format = "<H" if major == 1 else "<I"
        header_len = struct.unpack(header_len_format, f.read(struct.calcsize(header_len_format)))[0]
        header = ast.literal_eval(f.read(header_len).decode("latin1"))
    return tuple(int(dim) for dim in header["shape"])


def build_feature_config(raw_dir: Path, out_dir: Path, item_count: int) -> dict[str, object]:
    """记录官方多模态特征路径和维度，供后续 Dataset / Encoder 统一读取。"""
    feature_dir = raw_dir / FEATURE_DIR_NAME
    features: dict[str, dict[str, object]] = {}

    for feature_name, (stem, expected_dim) in FEATURE_SPECS.items():
        npy_path = feature_dir / f"{stem}.npy"
        json_path = feature_dir / f"{stem}.json"
        require_file(npy_path)
        require_file(json_path)

        shape = parse_npy_shape(npy_path)
        if shape != (item_count, expected_dim):
            raise ValueError(f"{feature_name} shape mismatch: expected={(item_count, expected_dim)}, actual={shape}")

        features[feature_name] = {
            "npy_path": project_relative(npy_path),
            "json_path": project_relative(json_path),
            "dim": expected_dim,
            "shape": list(shape),
        }

    return {
        "dataset": "microlens_100k",
        "feature_row_order": "ascending_item_id",
        "item_ids_path": project_relative(out_dir / "item_ids.csv"),
        "item_meta_path": project_relative(out_dir / "item_meta.csv"),
        "features": features,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """统一 CSV 写出逻辑，保证字段顺序稳定。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_item_meta(out_dir: Path, item_ids: list[int], titles: dict[int, str], likes_views: dict[int, tuple[int, int]]) -> None:
    """保存 item 文本和热度元信息，category 暂无时统一置为 __UNK__。"""
    rows = []
    for item_id in item_ids:
        likes, views = likes_views[item_id]
        rows.append(
            {
                "item_id": item_id,
                "title": titles[item_id],
                "likes": likes,
                "views": views,
                "category_id": UNKNOWN_CATEGORY,
            }
        )
    write_csv(out_dir / "item_meta.csv", rows, ["item_id", "title", "likes", "views", "category_id"])


def save_item_ids(out_dir: Path, item_ids: list[int]) -> None:
    """固定 item_id 到特征矩阵行号的映射；官方特征按 item_id 升序排列。"""
    rows = [{"item_index": idx, "item_id": item_id} for idx, item_id in enumerate(item_ids)]
    write_csv(out_dir / "item_ids.csv", rows, ["item_index", "item_id"])


def save_behavior_samples(out_dir: Path, samples: list[dict[str, object]], source_name: str) -> dict[str, int]:
    """保存全量样本和三份时间切分样本。"""
    fieldnames = [
        "user_id",
        "item_id",
        "timestamp",
        "sort_key",
        "label",
        "sample_weight",
        "category_id",
        "source_name",
        "split",
    ]
    rows = [{**row, "source_name": source_name} for row in samples]
    write_csv(out_dir / "behavior_samples_all.csv", rows, fieldnames)

    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_counts[split] = len(split_rows)
        write_csv(out_dir / f"behavior_samples_{split}.csv", split_rows, fieldnames)

    return split_counts


def save_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    raw_dir = resolve_project_path(args.raw_dir)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V3 Step 01 | Prepare MicroLens-100K Samples")
    print("=" * 80)
    print(f"[INFO] raw dir: {raw_dir}")
    print(f"[INFO] out dir: {out_dir}")

    title_path = raw_dir / TITLE_NAME
    likes_views_path = raw_dir / LIKES_VIEWS_NAME
    pairs_path = raw_dir / PAIRS_NAME
    for path in (title_path, likes_views_path, pairs_path):
        require_file(path)

    titles = load_titles(title_path)
    likes_views = load_likes_views(likes_views_path)
    item_ids = sorted(set(titles) & set(likes_views))
    if len(item_ids) != len(titles) or len(item_ids) != len(likes_views):
        raise ValueError("Title items and likes/views items are not perfectly aligned.")
    if item_ids != list(range(1, len(item_ids) + 1)):
        raise ValueError("MicroLens feature row order assumes contiguous item ids from 1 to N.")

    interactions = load_interactions(pairs_path, valid_item_ids=set(item_ids))
    samples, drop_stats = split_by_user(
        interactions=interactions,
        min_user_interactions=args.min_user_interactions,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    if not samples:
        raise ValueError("No behavior samples left after user filtering.")

    feature_config = build_feature_config(raw_dir, out_dir, item_count=len(item_ids))
    save_item_ids(out_dir, item_ids)
    save_item_meta(out_dir, item_ids, titles, likes_views)
    split_counts = save_behavior_samples(out_dir, samples, source_name=args.source_name)
    save_json(out_dir / "feature_config.json", feature_config)

    summary = {
        "dataset": "microlens_100k",
        "raw_dir": project_relative(raw_dir),
        "out_dir": project_relative(out_dir),
        "num_items": len(item_ids),
        "num_raw_interactions": len(interactions),
        "num_samples": len(samples),
        "num_users": len({int(row["user_id"]) for row in samples}),
        "split_counts": split_counts,
        "min_user_interactions": args.min_user_interactions,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "dropped_users": drop_stats["dropped_users"],
        "dropped_interactions": drop_stats["dropped_interactions"],
        "empty_titles_filled": sum(1 for title in titles.values() if title == NO_TITLE),
        "implicit_feedback_note": "All observed interactions are saved as label=1; negatives should be sampled during training.",
    }
    save_json(out_dir / "prepare_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Saved] {out_dir}")
    print("=" * 80)
    print("Prepare MicroLens-100K samples done.")


if __name__ == "__main__":
    main()
