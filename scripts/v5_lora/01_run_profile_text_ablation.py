"""
V5-LoRA Step 01: 运行 profile_text A-D 消融。

输入：
    V5 clean profiles + item title + MicroLens train/test 行为样本。

输出：
    A-D 四组 profile_text、对应 embedding，以及 full-catalog 召回指标。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.eval.content_recall import evaluate_content_recall
from src.v5.profile.io import load_item_meta, load_yaml, read_jsonl, save_csv, write_json
from src.v5.profile.paths import project_relative, resolve_project_path
from src.v5_lora.profile.profile_text import ABLATION_METHODS, build_ablation_text


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5_lora" / "profile_text_ablation.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V5-LoRA profile text ablation.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--methods", type=str, default="", help="Comma-separated methods from A-D.")
    parser.add_argument("--eval-split", type=str, default="")
    parser.add_argument("--max-items", type=int, default=-1)
    parser.add_argument("--max-eval-users", type=int, default=None)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_profiles(path: Path) -> dict[int, dict]:
    rows = read_jsonl(path)
    return {int(row["item_id"]): row["profile"] for row in rows}


def build_text_rows(method: str, item_meta: list[dict], profiles: dict[int, dict], max_items: int) -> list[dict]:
    rows = []
    for item in item_meta:
        item_id = int(item["item_id"])
        if item_id not in profiles:
            continue
        rows.append(
            {
                "item_id": item_id,
                "title": item["title"],
                "profile_text": build_ablation_text(method, item["title"], profiles[item_id]),
            }
        )
        if max_items > 0 and len(rows) >= max_items:
            break
    return rows


def load_embedding_model(cfg: dict):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Profile text ablation requires sentence-transformers.") from exc

    emb_cfg = cfg["embedding"]
    device = resolve_device(str(emb_cfg.get("device", "auto")))
    model = SentenceTransformer(emb_cfg["model_name"], device=device)
    model.max_seq_length = int(emb_cfg.get("max_seq_length", 512))
    return model


def encode_rows(rows: list[dict], cfg: dict, output_dir: Path, method: str, model) -> tuple[list[int], "np.ndarray"]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Profile text ablation requires numpy.") from exc

    emb_cfg = cfg["embedding"]
    item_ids = [int(row["item_id"]) for row in rows]
    texts = [row["profile_text"] for row in rows]
    embeddings = model.encode(
        texts,
        batch_size=int(emb_cfg["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
    ).astype("float32")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{method}_item_ids.npy", np.asarray(item_ids, dtype="int64"))
    np.save(output_dir / f"{method}_embeddings.npy", embeddings)
    return item_ids, embeddings


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or list(cfg["ablation"]["methods"])
    for method in methods:
        if method not in ABLATION_METHODS:
            raise ValueError(f"Unsupported method: {method}")

    item_meta = load_item_meta(resolve_project_path(cfg["data"]["item_meta"]))
    profiles = load_profiles(resolve_project_path(cfg["data"]["profiles_clean"]))
    text_dir = resolve_project_path(cfg["output"]["text_dir"])
    embedding_dir = resolve_project_path(cfg["output"]["embedding_dir"])
    eval_split = args.eval_split or cfg["ablation"].get("eval_split", "test")
    eval_path = resolve_project_path(cfg["data"][f"{eval_split}_samples"])
    train_path = resolve_project_path(cfg["data"]["train_samples"])
    ks = [int(k) for k in cfg["ablation"]["ks"]]
    max_eval_users_value = (
        args.max_eval_users
        if args.max_eval_users is not None
        else int(cfg["ablation"].get("max_eval_users", -1))
    )
    max_eval_users = None if max_eval_users_value == -1 else max_eval_users_value

    report = {"eval_split": eval_split, "methods": {}}
    model = load_embedding_model(cfg)
    for method in methods:
        rows = build_text_rows(method, item_meta, profiles, args.max_items)
        text_path = text_dir / f"{method}.csv"
        save_csv(text_path, rows, ["item_id", "title", "profile_text"])
        item_ids, embeddings = encode_rows(rows, cfg, embedding_dir, method, model)
        metrics = evaluate_content_recall(
            item_vectors=embeddings,
            item_ids=item_ids,
            train_path=train_path,
            eval_path=eval_path,
            ks=ks,
            max_history_len=int(cfg["ablation"]["max_history_len"]),
            max_eval_users=max_eval_users,
            user_batch_size=int(cfg["ablation"].get("user_batch_size", 256)),
        )
        report["methods"][method] = metrics
        print(f"[{method}] {json.dumps(metrics, ensure_ascii=False)}")

    output = resolve_project_path(cfg["output"]["report"])
    write_json(output, report)
    print("=" * 80)
    print("V5-LoRA Step 01: Profile Text Ablation")
    print("=" * 80)
    print(f"text_dir: {project_relative(text_dir)}")
    print(f"embedding_dir: {project_relative(embedding_dir)}")
    print(f"report: {project_relative(output)}")


if __name__ == "__main__":
    main()
