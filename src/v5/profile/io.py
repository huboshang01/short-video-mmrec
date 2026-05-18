"""V5 文件读写工具。

这些函数只处理轻量 CSV / JSONL，不绑定具体模型，方便脚本复用。
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
from collections.abc import Iterable


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSONL.") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_item_meta(path: Path) -> list[dict]:
    """读取 V3 item_meta.csv，并把常用数值字段转成 int。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["item_id"] = int(row["item_id"])
        row["likes"] = int(float(row.get("likes") or 0))
        row["views"] = int(float(row.get("views") or 0))
        row["title"] = (row.get("title") or "").strip()
    return rows


def load_item_ids(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [int(row["item_id"]) for row in csv.DictReader(f)]


def save_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
