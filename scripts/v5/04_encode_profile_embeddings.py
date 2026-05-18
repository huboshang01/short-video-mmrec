"""
V5 Step 04: 将 profile_text.csv 编码为 profile embedding。
这一步和V1 的03_encode_video_text.py一致，但V5 是编码 MLLM 生成的 profile_text。

默认使用 BGE-M3；输出 item_ids.npy + embeddings.npy，供 FAISS 和召回评估复用。
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

from src.v5.profile.io import load_yaml
from src.v5.profile.paths import project_relative, resolve_project_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5" / "profile_embedding.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode V5 profile texts into embeddings.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=-1)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Profile embedding requires numpy and sentence-transformers.") from exc

    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    emb_cfg = cfg["embedding"]
    input_path = resolve_project_path(args.input or cfg["data"]["profile_text"])
    model_name = args.model_name or emb_cfg["model_name"]
    batch_size = args.batch_size or int(emb_cfg["batch_size"])

    rows = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]
    item_ids = np.asarray([int(row["item_id"]) for row in rows], dtype="int64")
    texts = [row["profile_text"] for row in rows]

    device = resolve_device(str(emb_cfg.get("device", "auto")))
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = int(emb_cfg.get("max_seq_length", 512))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
    ).astype("float32")

    ids_path = resolve_project_path(cfg["output"]["item_ids"])
    emb_path = resolve_project_path(cfg["output"]["embeddings"])
    config_path = resolve_project_path(cfg["output"]["config"])
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, item_ids)
    np.save(emb_path, embeddings)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "device": device,
                "num_items": int(len(item_ids)),
                "embedding_dim": int(embeddings.shape[1]),
                "normalized": bool(emb_cfg.get("normalize_embeddings", True)),
                "source": project_relative(input_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print("=" * 80)
    print("V5 Step 04: Encode Profile Embeddings")
    print("=" * 80)
    print(f"items: {len(item_ids)}")
    print(f"embeddings: {project_relative(emb_path)} {embeddings.shape}")
    print(f"ids: {project_relative(ids_path)}")
    print(f"config: {project_relative(config_path)}")


if __name__ == "__main__":
    main()
