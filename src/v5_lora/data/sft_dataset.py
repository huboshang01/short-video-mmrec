"""LLaMA-Factory 多图 SFT 数据构造工具。"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.v5_lora.profile.prompt import build_lora_profile_prompt


def build_sft_row(item: dict, profile: dict, system_prompt: str = "") -> dict:
    """把一个 item 转成 LLaMA-Factory Alpaca 多模态样本。

    输入：
        item: V5 profile_inputs 中的一行，包含 title 和 frames。
        profile: retrieval-aware teacher profile。

    输出：
        instruction/input/output/images 格式，images 数量与 <image> 标签数量一致。
    """
    frames = list(item["frames"])
    instruction = "\n".join(["<image>" for _ in frames])
    instruction += "\n" + build_lora_profile_prompt(item["title"])
    row = {
        "instruction": instruction,
        "input": "",
        "output": json.dumps(profile, ensure_ascii=False),
        "images": frames,
    }
    if system_prompt:
        row["system"] = system_prompt
    return row


def split_rows(rows: list[dict], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """按 item 维度随机切分 train/val/test。"""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    train_end = int(len(rows) * train_ratio)
    val_end = train_end + int(len(rows) * val_ratio)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def write_dataset_info(path: Path, names: dict[str, str]) -> None:
    """写 LLaMA-Factory dataset_info.json。"""
    info = {}
    for dataset_name, file_name in names.items():
        info[dataset_name] = {
            "file_name": file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "images": "images",
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
        f.write("\n")
