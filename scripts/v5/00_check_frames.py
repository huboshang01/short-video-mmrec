"""
V5 Step 00: 检查 MicroLens-100K 5 frames，并生成 frame_manifest.csv。

manifest 是 V5 后续所有多图输入的基础：一行一个 item，记录该 item 的帧路径。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.profile.frames import build_manifest_rows, discover_frames
from src.v5.profile.io import load_item_ids, load_yaml, save_csv, write_json
from src.v5.profile.paths import project_relative, resolve_project_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5" / "profile_generation.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MicroLens-100K 5-frame files for V5.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--frames-dir", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    frames_dir = resolve_project_path(args.frames_dir or cfg["data"]["frames_dir"])
    output = resolve_project_path(args.output or cfg["data"]["frame_manifest"])
    item_ids = load_item_ids(resolve_project_path(cfg["data"]["item_ids"]))
    expected = int(cfg["frames"]["expected_frames_per_item"])
    allowed = {ext.lower() for ext in cfg["frames"]["allowed_extensions"]}

    frames_by_item = discover_frames(frames_dir, allowed_extensions=allowed)
    rows, summary = build_manifest_rows(item_ids, frames_by_item, expected_frames=expected, root=PROJECT_ROOT)
    save_csv(output, rows, ["item_id", "frame_count", "is_complete", "frame_paths"])
    summary_path = output.with_suffix(".summary.json")
    write_json(summary_path, summary)

    print("=" * 80)
    print("V5 Step 00: Frame Check")
    print("=" * 80)
    print(f"frames_dir: {project_relative(frames_dir)}")
    print(f"manifest: {project_relative(output)}")
    print(f"summary: {project_relative(summary_path)}")
    print(summary)


if __name__ == "__main__":
    main()
