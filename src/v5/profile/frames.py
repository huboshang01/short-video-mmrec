"""MicroLens-100K 5 frames 发现与 manifest 构造。"""

from __future__ import annotations

from pathlib import Path
import csv
import re


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def infer_item_id(path: Path) -> int | None:
    """尽量兼容常见解压结构：item_id 目录，或 item_id_xxx.jpg 文件名。"""
    for part in [path.parent.name, path.stem]:
        if part.isdigit():
            return int(part)
        match = re.match(r"^(\d+)(?:[_\-.].*)?$", part)
        if match:
            return int(match.group(1))
    return None


def discover_frames(frames_dir: Path, allowed_extensions: set[str] | None = None) -> dict[int, list[Path]]:
    allowed = allowed_extensions or IMAGE_EXTENSIONS
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    grouped: dict[int, list[Path]] = {}
    for path in frames_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        item_id = infer_item_id(path)
        if item_id is None:
            continue
        grouped.setdefault(item_id, []).append(path)

    for paths in grouped.values():
        paths.sort(key=lambda p: str(p))
    return grouped


def build_manifest_rows(
    item_ids: list[int],
    frames_by_item: dict[int, list[Path]],
    expected_frames: int,
    root: Path,
) -> tuple[list[dict], dict]:
    """输出一行一个 item，frame_paths 用 | 连接，便于人工查看和脚本读取。"""
    rows: list[dict] = []
    complete = 0
    root = root.resolve()
    for item_id in item_ids:
        frames = frames_by_item.get(int(item_id), [])[:expected_frames]
        paths = []
        for path in frames:
            resolved = path.resolve()
            try:
                paths.append(str(resolved.relative_to(root)))
            except ValueError:
                paths.append(str(resolved))
        is_complete = len(paths) >= expected_frames
        complete += int(is_complete)
        rows.append(
            {
                "item_id": int(item_id),
                "frame_count": len(paths),
                "is_complete": int(is_complete),
                "frame_paths": "|".join(paths),
            }
        )
    summary = {
        "num_items": len(item_ids),
        "items_with_frames": sum(1 for row in rows if int(row["frame_count"]) > 0),
        "complete_items": complete,
        "expected_frames_per_item": expected_frames,
    }
    return rows, summary


def load_frame_manifest(path: Path, root: Path) -> dict[int, list[Path]]:
    """读取 manifest，并把项目内相对路径还原成 Path。"""
    manifest: dict[int, list[Path]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            paths = [root / p for p in row.get("frame_paths", "").split("|") if p]
            manifest[int(row["item_id"])] = paths
    return manifest
