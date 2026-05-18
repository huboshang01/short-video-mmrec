"""
V5-LoRA Step 03: 构造 LLaMA-Factory 多图 SFT 数据。

输入：
    V5 profile_inputs.jsonl：提供 title 和 5 frames。
    retrieval_aware_teacher_profiles.jsonl：提供 LoRA 监督答案。

输出：
    sft_train.json / sft_val.json / sft_test.json / dataset_info.json。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.profile.io import load_yaml, read_jsonl, write_json
from src.v5.profile.paths import project_relative, resolve_project_path
from src.v5_lora.data.sft_dataset import build_sft_row, split_rows, write_dataset_info


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5_lora" / "sft_dataset.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLaMA-Factory SFT dataset for V5-LoRA.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def write_json_array(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    profile_inputs = {int(row["item_id"]): row for row in read_jsonl(resolve_project_path(cfg["data"]["profile_inputs"]))}
    teacher_rows = read_jsonl(resolve_project_path(cfg["data"]["teacher_profiles"]))
    max_items = args.max_items if args.max_items is not None else int(cfg["split"].get("max_items", -1))

    rows = []
    for teacher in teacher_rows:
        item_id = int(teacher["item_id"])
        if item_id not in profile_inputs:
            continue
        rows.append(
            build_sft_row(
                item=profile_inputs[item_id],
                profile=teacher["profile"],
                system_prompt=cfg["dataset"].get("system_prompt", ""),
            )
        )
        if max_items > 0 and len(rows) >= max_items:
            break

    train_rows, val_rows, test_rows = split_rows(
        rows,
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        seed=int(cfg["split"]["seed"]),
    )
    output_dir = resolve_project_path(cfg["data"]["output_dir"])
    files = {
        "train": "sft_train.json",
        "val": "sft_val.json",
        "test": "sft_test.json",
    }
    write_json_array(output_dir / files["train"], train_rows)
    write_json_array(output_dir / files["val"], val_rows)
    write_json_array(output_dir / files["test"], test_rows)

    prefix = cfg["dataset"].get("name_prefix", "v5_lora")
    dataset_info = resolve_project_path(cfg["data"]["dataset_info"])
    write_dataset_info(
        dataset_info,
        {
            f"{prefix}_train": files["train"],
            f"{prefix}_val": files["val"],
            f"{prefix}_test": files["test"],
        },
    )
    write_json(
        output_dir / "sft_dataset.summary.json",
        {"total": len(rows), "train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
    )

    print("=" * 80)
    print("V5-LoRA Step 03: Build LLaMA-Factory SFT Dataset")
    print("=" * 80)
    print(f"output_dir: {project_relative(output_dir)}")
    print(f"dataset_info: {project_relative(dataset_info)}")
    print({"total": len(rows), "train": len(train_rows), "val": len(val_rows), "test": len(test_rows)})


if __name__ == "__main__":
    main()
