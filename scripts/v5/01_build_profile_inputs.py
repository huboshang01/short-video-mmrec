"""
V5 Step 01: 构造 MLLM profile 生成输入。

输入来自 item_meta.csv + frame_manifest.csv；输出 JSONL，每行代表一个短视频 item。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.profile.frames import load_frame_manifest
from src.v5.profile.io import load_item_meta, load_yaml, write_json, write_jsonl
from src.v5.profile.paths import project_relative, resolve_project_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5" / "profile_generation.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V5 MLLM profile input JSONL.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    item_meta = load_item_meta(resolve_project_path(cfg["data"]["item_meta"]))
    manifest = load_frame_manifest(resolve_project_path(cfg["data"]["frame_manifest"]), root=PROJECT_ROOT)
    output = resolve_project_path(args.output or cfg["data"]["profile_inputs"])
    expected = int(cfg["frames"]["expected_frames_per_item"])
    allow_incomplete = args.allow_incomplete or bool(cfg["frames"].get("allow_incomplete_items", False))
    max_items = args.max_items if args.max_items is not None else int(cfg["generation"].get("max_items", -1))

    rows = []
    skipped = 0
    for item in item_meta:
        frames = manifest.get(int(item["item_id"]), [])
        if len(frames) < expected and not allow_incomplete:
            skipped += 1
            continue
        rows.append(
            {
                "item_id": int(item["item_id"]),
                "title": item["title"],
                "likes": int(item["likes"]),
                "views": int(item["views"]),
                "category_id": item.get("category_id", "__UNK__"),
                "frames": [project_relative(path) for path in frames[:expected]],
            }
        )
        if max_items and max_items > 0 and len(rows) >= max_items:
            break

    count = write_jsonl(output, rows)
    summary = {"output_rows": count, "skipped_incomplete_items": skipped, "expected_frames_per_item": expected}
    write_json(output.with_suffix(".summary.json"), summary)
    print("=" * 80)
    print("V5 Step 01: Build Profile Inputs")
    print("=" * 80)
    print(f"output: {project_relative(output)}")
    print(summary)


if __name__ == "__main__":
    main()
