"""
V5-LoRA Step 00: 检查输入文件与关键行数。

该脚本不生成新数据，只确认 V5-LoRA 依赖的 V5 profile、frames、item_meta
和行为评估样本已经准备好。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.profile.io import load_yaml, read_jsonl
from src.v5.profile.paths import project_relative, resolve_project_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5_lora" / "profile_retrieval_lora.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check V5-LoRA input files.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    paths = {
        "item_meta": cfg["data"]["item_meta"],
        "item_ids": cfg["data"]["item_ids"],
        "train_samples": cfg["data"]["train_samples"],
        "test_samples": cfg["data"]["test_samples"],
        "feature_config": cfg["data"]["feature_config"],
        "v5_profiles_clean": "data/processed/v5/microlens_100k/profiles_clean.jsonl",
        "v5_profile_inputs": "data/processed/v5/microlens_100k/profile_inputs.jsonl",
    }

    report = {}
    missing = []
    for name, raw_path in paths.items():
        path = resolve_project_path(raw_path)
        exists = path.exists()
        if not exists:
            missing.append(name)
        report[name] = {"path": project_relative(path), "exists": exists}
        if exists and path.suffix == ".csv":
            report[name]["rows"] = count_csv_rows(path)
        if exists and path.suffix == ".jsonl":
            report[name]["rows"] = len(read_jsonl(path))

    print("=" * 80)
    print("V5-LoRA Step 00: Check Inputs")
    print("=" * 80)
    for name, info in report.items():
        rows = f", rows={info['rows']}" if "rows" in info else ""
        print(f"{name}: {info['path']} exists={info['exists']}{rows}")
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")


if __name__ == "__main__":
    main()
