"""
V3 Step 00: MicroLens-100K 原始数据检查。

本脚本只做“读数据、验结构、查对齐”，不生成训练样本：
    1. 检查 MicroLens-100K 必需文件是否存在。
    2. 统计 interaction / title / likes_views 的规模和 item 覆盖关系。
    3. 解析官方预提取多模态 .npy 文件头，确认 shape / dtype / 文件大小。
    4. 流式统计大 JSON 特征文件的 item key 数量，避免一次性加载数百 MB JSON。

设计原则：
    - 只依赖 Python 标准库，方便在新环境中先做数据体检。
    - 对关键一致性问题直接报错，避免后续训练阶段才发现特征错位。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import ast
import csv
import json
import math
import re
import struct


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "microlens_100k"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "v3" / "microlens_100k" / "raw_check_summary.json"

PAIRS_NAME = "MicroLens-100k_pairs.csv"
TITLE_NAME = "MicroLens-100k_title_en.csv"
LIKES_VIEWS_NAME = "MicroLens-100k_likes_and_views.txt"
FEATURE_DIR_NAME = "extracted_modality_features"

FEATURE_SPECS = {
    "text": ("MicroLens-100k_title_en_text_features_BgeM3", 1024),
    "image": ("MicroLens-100k_image_features_CLIPRN50", 1024),
    "video": ("MicroLens-100k_video_features_VideoMAE", 768),
}


@dataclass(frozen=True)
class InteractionStats:
    """MicroLens interaction 表的核心统计。"""

    rows: int
    num_users: int
    item_ids: set[int]
    min_timestamp: int
    max_timestamp: int
    duplicate_exact_rows: int


@dataclass(frozen=True)
class TitleStats:
    """标题表统计，保留 item 集合用于跨文件对齐检查。"""

    rows: int
    item_ids: set[int]
    empty_titles: int


@dataclass(frozen=True)
class LikesViewsStats:
    """likes/views 热度表统计。"""

    rows: int
    item_ids: set[int]
    min_likes: int
    max_likes: int
    min_views: int
    max_views: int


@dataclass(frozen=True)
class NpyInfo:
    """从 .npy header 解析出的轻量元信息。"""

    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool
    expected_bytes: int
    actual_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MicroLens-100K raw data for V3.")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Path to MicroLens-100K raw directory.")
    parser.add_argument("--summary-output", type=str, default=str(DEFAULT_SUMMARY_PATH), help="Path to save check summary JSON.")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    """支持绝对路径和相对项目根目录的路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path) -> None:
    """必需文件缺失时立即失败，避免后续统计产生误导。"""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_raw_files(raw_dir: Path) -> dict[str, Path]:
    """集中声明并检查 V3 MVP 依赖的 MicroLens 文件。"""
    feature_dir = raw_dir / FEATURE_DIR_NAME
    paths = {
        "pairs": raw_dir / PAIRS_NAME,
        "title": raw_dir / TITLE_NAME,
        "likes_views": raw_dir / LIKES_VIEWS_NAME,
        "feature_dir": feature_dir,
    }

    for key, path in paths.items():
        if key == "feature_dir":
            if not path.is_dir():
                raise FileNotFoundError(f"Required feature directory not found: {path}")
        else:
            require_file(path)

    for feature_name, (stem, _) in FEATURE_SPECS.items():
        paths[f"{feature_name}_npy"] = feature_dir / f"{stem}.npy"
        paths[f"{feature_name}_json"] = feature_dir / f"{stem}.json"
        require_file(paths[f"{feature_name}_npy"])
        require_file(paths[f"{feature_name}_json"])

    return paths


def read_interaction_stats(path: Path) -> InteractionStats:
    """读取 user-item-time 交互；MicroLens-100K CSV 有表头 user,item,timestamp。"""
    rows = 0
    users: set[int] = set()
    item_ids: set[int] = set()
    seen_rows: set[tuple[int, int, int]] = set()
    duplicate_exact_rows = 0
    min_timestamp: int | None = None
    max_timestamp: int | None = None

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

            rows += 1
            users.add(user_id)
            item_ids.add(item_id)
            min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
            max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)

            key = (user_id, item_id, timestamp)
            if key in seen_rows:
                duplicate_exact_rows += 1
            else:
                seen_rows.add(key)

    if rows == 0:
        raise ValueError(f"No interactions found in {path}")

    return InteractionStats(
        rows=rows,
        num_users=len(users),
        item_ids=item_ids,
        min_timestamp=int(min_timestamp),
        max_timestamp=int(max_timestamp),
        duplicate_exact_rows=duplicate_exact_rows,
    )


def read_title_stats(path: Path) -> TitleStats:
    """读取无表头标题文件；标题里若出现逗号，保守地把第 2 列之后重新拼回文本。"""
    rows = 0
    item_ids: set[int] = set()
    empty_titles = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            item_id = int(row[0].strip())
            title = ",".join(row[1:]).strip()

            rows += 1
            item_ids.add(item_id)
            if not title:
                empty_titles += 1

    if rows == 0:
        raise ValueError(f"No titles found in {path}")
    if rows != len(item_ids):
        raise ValueError(f"Duplicated item ids found in title file: rows={rows}, unique={len(item_ids)}")

    return TitleStats(rows=rows, item_ids=item_ids, empty_titles=empty_titles)


def read_likes_views_stats(path: Path) -> LikesViewsStats:
    """读取 tab 分隔的 item 热度表：item_id, likes, views。"""
    rows = 0
    item_ids: set[int] = set()
    min_likes: int | None = None
    max_likes: int | None = None
    min_views: int | None = None
    max_views: int | None = None

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no} expected 3 tab-separated fields, got {len(parts)}")

            item_id, likes, views = int(parts[0]), int(parts[1]), int(parts[2])
            rows += 1
            item_ids.add(item_id)
            min_likes = likes if min_likes is None else min(min_likes, likes)
            max_likes = likes if max_likes is None else max(max_likes, likes)
            min_views = views if min_views is None else min(min_views, views)
            max_views = views if max_views is None else max(max_views, views)

    if rows == 0:
        raise ValueError(f"No likes/views rows found in {path}")
    if rows != len(item_ids):
        raise ValueError(f"Duplicated item ids found in likes/views file: rows={rows}, unique={len(item_ids)}")

    return LikesViewsStats(
        rows=rows,
        item_ids=item_ids,
        min_likes=int(min_likes),
        max_likes=int(max_likes),
        min_views=int(min_views),
        max_views=int(max_views),
    )


def parse_npy_header(path: Path) -> NpyInfo:
    """只解析 .npy header，不加载整个矩阵到内存。"""
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not a valid .npy file.")

        major, _minor = f.read(2)
        header_len_format = "<H" if major == 1 else "<I"
        header_len_size = struct.calcsize(header_len_format)
        header_len = struct.unpack(header_len_format, f.read(header_len_size))[0]
        header = ast.literal_eval(f.read(header_len).decode("latin1"))

    shape = tuple(int(dim) for dim in header["shape"])
    dtype = str(header["descr"])
    fortran_order = bool(header["fortran_order"])
    item_size = dtype_item_size(dtype)
    header_bytes = 6 + 2 + header_len_size + header_len
    expected_bytes = header_bytes + math.prod(shape) * item_size
    actual_bytes = path.stat().st_size

    if expected_bytes != actual_bytes:
        raise ValueError(f"{path} size mismatch: expected={expected_bytes}, actual={actual_bytes}")

    return NpyInfo(
        shape=shape,
        dtype=dtype,
        fortran_order=fortran_order,
        expected_bytes=expected_bytes,
        actual_bytes=actual_bytes,
    )


def dtype_item_size(dtype: str) -> int:
    """解析常见 numpy dtype 描述，例如 '<f4' 表示 float32。"""
    match = re.search(r"(\d+)$", dtype)
    if not match:
        raise ValueError(f"Cannot infer dtype item size from {dtype!r}")
    return int(match.group(1))


def read_json_item_keys(path: Path) -> set[int]:
    """流式读取形如 \"123\": 的顶层 item key；集合去重可自然处理 chunk 重叠。"""
    pattern = re.compile(r'"(\d+)"\s*:')
    keys: set[int] = set()
    tail = ""

    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break

            data = tail + chunk
            keys.update(int(key) for key in pattern.findall(data))
            # 保留一小段尾部，确保跨 chunk 的 key 也能在下一轮被完整匹配。
            tail = data[-64:]

    keys.update(int(key) for key in pattern.findall(tail))
    return keys


def check_item_alignment(
    interaction_stats: InteractionStats,
    title_stats: TitleStats,
    likes_stats: LikesViewsStats,
) -> dict[str, int]:
    """检查三张 item 相关表是否覆盖同一批 item。"""
    return {
        "interaction_items_missing_title": len(interaction_stats.item_ids - title_stats.item_ids),
        "title_items_missing_interactions": len(title_stats.item_ids - interaction_stats.item_ids),
        "likes_items_missing_title": len(likes_stats.item_ids - title_stats.item_ids),
        "title_items_missing_likes": len(title_stats.item_ids - likes_stats.item_ids),
    }


def check_features(paths: dict[str, Path], title_item_ids: set[int]) -> dict[str, dict[str, object]]:
    """检查 text/image/video 三类官方预提取特征的 shape 和 JSON key 数。"""
    feature_summary: dict[str, dict[str, object]] = {}
    num_items = len(title_item_ids)

    for feature_name, (_stem, expected_dim) in FEATURE_SPECS.items():
        npy_info = parse_npy_header(paths[f"{feature_name}_npy"])
        json_item_ids = read_json_item_keys(paths[f"{feature_name}_json"])

        if npy_info.shape != (num_items, expected_dim):
            raise ValueError(
                f"{feature_name} feature shape mismatch: "
                f"expected={(num_items, expected_dim)}, actual={npy_info.shape}"
            )
        if json_item_ids != title_item_ids:
            raise ValueError(
                f"{feature_name} JSON item ids do not match title item ids: "
                f"missing={len(title_item_ids - json_item_ids)}, extra={len(json_item_ids - title_item_ids)}"
            )
        if npy_info.dtype != "<f4":
            raise ValueError(f"{feature_name} feature dtype must be '<f4', got {npy_info.dtype}")
        if npy_info.fortran_order:
            raise ValueError(f"{feature_name} feature must be C-contiguous, got fortran_order=True")

        feature_summary[feature_name] = {
            "npy_path": str(paths[f"{feature_name}_npy"].relative_to(PROJECT_ROOT)),
            "json_path": str(paths[f"{feature_name}_json"].relative_to(PROJECT_ROOT)),
            "shape": list(npy_info.shape),
            "dtype": npy_info.dtype,
            "json_key_count": len(json_item_ids),
            "bytes": npy_info.actual_bytes,
        }

    return feature_summary


def save_summary(summary: dict[str, object], output_path: Path) -> None:
    """保存检查摘要，作为后续 processed 数据生成的可追溯记录。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    raw_dir = resolve_project_path(args.raw_dir)
    summary_output = resolve_project_path(args.summary_output)

    print("=" * 80)
    print("V3 Step 00 | Check MicroLens-100K Raw Data")
    print("=" * 80)
    print(f"[INFO] raw dir: {raw_dir}")

    paths = require_raw_files(raw_dir)
    interaction_stats = read_interaction_stats(paths["pairs"])
    title_stats = read_title_stats(paths["title"])
    likes_stats = read_likes_views_stats(paths["likes_views"])

    alignment = check_item_alignment(interaction_stats, title_stats, likes_stats)
    if any(value != 0 for value in alignment.values()):
        raise ValueError(f"Item alignment check failed: {alignment}")

    feature_summary = check_features(paths, title_item_ids=title_stats.item_ids)
    summary = {
        "raw_dir": str(raw_dir.relative_to(PROJECT_ROOT)),
        "interactions": {
            "rows": interaction_stats.rows,
            "num_users": interaction_stats.num_users,
            "num_items": len(interaction_stats.item_ids),
            "timestamp_min": interaction_stats.min_timestamp,
            "timestamp_max": interaction_stats.max_timestamp,
            "duplicate_exact_rows": interaction_stats.duplicate_exact_rows,
        },
        "titles": {
            "rows": title_stats.rows,
            "num_items": len(title_stats.item_ids),
            "empty_titles": title_stats.empty_titles,
        },
        "likes_views": {
            "rows": likes_stats.rows,
            "num_items": len(likes_stats.item_ids),
            "likes_min": likes_stats.min_likes,
            "likes_max": likes_stats.max_likes,
            "views_min": likes_stats.min_views,
            "views_max": likes_stats.max_views,
        },
        "alignment": alignment,
        "features": feature_summary,
    }

    save_summary(summary, summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Saved] {summary_output}")
    print("=" * 80)
    print("MicroLens-100K raw data check passed.")


if __name__ == "__main__":
    main()
