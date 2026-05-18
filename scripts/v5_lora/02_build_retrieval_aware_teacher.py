"""
V5-LoRA Step 02: 构造轻量 retrieval-aware teacher profile。

第一版只使用 title + V5 clean profile，不引入行为邻居和 hard negatives，
重点保留标题关键词、推荐检索词、兴趣标签等召回友好字段。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.profile.io import load_item_meta, load_yaml, read_jsonl, write_json, write_jsonl
from src.v5.profile.paths import project_relative, resolve_project_path
from src.v5_lora.profile.profile_text import build_lora_profile_text
from src.v5_lora.profile.schema import clean_lora_profile
from src.v5_lora.profile.teacher_profile import build_retrieval_aware_profile


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5_lora" / "teacher_profile.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval-aware teacher profiles for V5-LoRA.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    item_meta = {int(row["item_id"]): row for row in load_item_meta(resolve_project_path(cfg["data"]["item_meta"]))}
    rows = read_jsonl(resolve_project_path(cfg["data"]["profiles_clean"]))
    max_items = args.max_items if args.max_items is not None else int(cfg["teacher"].get("max_items", -1))

    teacher_rows = []
    text_rows = []
    report = {"total": 0, "valid": 0, "missing_field_rows": 0, "empty_list_rows": 0}
    for row in rows:
        item_id = int(row["item_id"])
        title = item_meta.get(item_id, {}).get("title", row.get("title", ""))
        profile = build_retrieval_aware_profile(title, row["profile"])
        cleaned, quality = clean_lora_profile(profile, title=title)
        teacher_rows.append({**row, "title": title, "profile": cleaned, "quality": quality, "source": "v5_lora_rule"})
        text_rows.append(
            {
                "item_id": item_id,
                "title": title,
                "profile_text": build_lora_profile_text(title, cleaned, mode=cfg["teacher"].get("profile_text_mode", "lora_full")),
            }
        )
        report["total"] += 1
        report["valid"] += int(bool(quality.get("is_valid")))
        report["missing_field_rows"] += int(bool(quality.get("missing_fields")))
        report["empty_list_rows"] += int(bool(quality.get("empty_list_fields")))
        if max_items > 0 and len(teacher_rows) >= max_items:
            break

    output = resolve_project_path(cfg["data"]["teacher_profiles"])
    text_output = resolve_project_path(cfg["data"]["profile_text"])
    write_jsonl(output, teacher_rows)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    with text_output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "title", "profile_text"])
        writer.writeheader()
        writer.writerows(text_rows)
    write_json(output.with_suffix(".summary.json"), report)

    print("=" * 80)
    print("V5-LoRA Step 02: Build Retrieval-Aware Teacher")
    print("=" * 80)
    print(f"teacher profiles: {project_relative(output)}")
    print(f"profile text: {project_relative(text_output)}")
    print(report)


if __name__ == "__main__":
    main()
