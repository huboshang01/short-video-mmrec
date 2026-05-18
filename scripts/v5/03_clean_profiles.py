"""
V5 Step 03: 清洗 MLLM 原始输出，并生成 profile_text.csv。

清洗目标：保证每个 profile 都是固定 schema 的合法 JSON，后续可稳定编码为 embedding。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.profile.io import load_yaml, read_jsonl, write_json, write_jsonl
from src.v5.profile.paths import project_relative, resolve_project_path
from src.v5.profile.schema import clean_profile, extract_json_object, profile_to_text


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5" / "profile_generation.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean V5 raw semantic profiles.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--text-output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    input_path = resolve_project_path(args.input or cfg["data"]["profiles_raw"])
    output_path = resolve_project_path(args.output or cfg["data"]["profiles_clean"])
    text_path = resolve_project_path(args.text_output or cfg["data"]["profile_text"])
    report_path = resolve_project_path(cfg["data"]["quality_report"])
    raw_rows = read_jsonl(input_path)

    clean_rows = []
    text_rows = []
    report = {"total": len(raw_rows), "valid": 0, "parse_failed": 0, "missing_field_rows": 0, "empty_list_rows": 0}

    for row in raw_rows:
        try:
            if isinstance(row.get("profile"), dict):
                profile = row["profile"]
            else:
                profile = extract_json_object(row.get("raw_response", ""))
            cleaned, quality = clean_profile(profile, title=row.get("title", ""))
        except (TypeError, ValueError) as exc:
            report["parse_failed"] += 1
            cleaned, quality = clean_profile({}, title=row.get("title", ""))
            quality["parse_error"] = str(exc)

        report["valid"] += int(bool(quality.get("is_valid")))
        report["missing_field_rows"] += int(bool(quality.get("missing_fields")))
        report["empty_list_rows"] += int(bool(quality.get("empty_list_fields")))
        clean_row = {**row, "profile": cleaned, "quality": quality}
        clean_row.pop("raw_response", None)
        clean_rows.append(clean_row)
        text_rows.append(
            {
                "item_id": int(row["item_id"]),
                "title": row.get("title", ""),
                "profile_text": profile_to_text(cleaned),
            }
        )

    write_jsonl(output_path, clean_rows)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with text_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "title", "profile_text"])
        writer.writeheader()
        writer.writerows(text_rows)
    write_json(report_path, report)

    print("=" * 80)
    print("V5 Step 03: Clean Profiles")
    print("=" * 80)
    print(f"clean profiles: {project_relative(output_path)}")
    print(f"profile text: {project_relative(text_path)}")
    print(f"quality report: {project_relative(report_path)}")
    print(report)


if __name__ == "__main__":
    main()
