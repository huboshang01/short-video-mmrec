"""
V5-LoRA Step 06: 清洗 LoRA profile 并回灌 V5 召回评估链路。

该脚本合并执行：raw profile 清洗、profile_text 构造、embedding、FAISS index、
以及 title/profile/multimodal/fusion full-catalog 召回评估。
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
from src.v5.profile.io import load_item_ids, load_item_meta, load_yaml, read_jsonl, write_json, write_jsonl
from src.v5.profile.paths import project_relative, resolve_project_path
from src.v5.profile.schema import extract_json_object
from src.v5.retrieval.embeddings import build_feature_matrix, load_feature_config
from src.v5_lora.profile.profile_text import build_lora_profile_text
from src.v5_lora.profile.schema import clean_lora_profile


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5_lora" / "profile_retrieval_lora.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V5-LoRA generated profiles.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--eval-split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--max-items", type=int, default=-1)
    parser.add_argument("--max-eval-users", type=int, default=None)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def clean_profiles(cfg: dict, max_items: int) -> tuple[list[int], list[dict]]:
    raw_rows = read_jsonl(resolve_project_path(cfg["data"]["profiles_raw"]))
    item_meta = {int(row["item_id"]): row for row in load_item_meta(resolve_project_path(cfg["data"]["item_meta"]))}
    clean_rows = []
    text_rows = []
    report = {"total": 0, "valid": 0, "parse_failed": 0, "missing_field_rows": 0, "empty_list_rows": 0}

    for row in raw_rows:
        title = item_meta.get(int(row["item_id"]), {}).get("title", row.get("title", ""))
        try:
            profile = row["profile"] if isinstance(row.get("profile"), dict) else extract_json_object(row.get("raw_response", ""))
            cleaned, quality = clean_lora_profile(profile, title=title)
        except (TypeError, ValueError) as exc:
            report["parse_failed"] += 1
            cleaned, quality = clean_lora_profile({}, title=title)
            quality["parse_error"] = str(exc)

        report["total"] += 1
        report["valid"] += int(bool(quality.get("is_valid")))
        report["missing_field_rows"] += int(bool(quality.get("missing_fields")))
        report["empty_list_rows"] += int(bool(quality.get("empty_list_fields")))
        clean_row = {**row, "title": title, "profile": cleaned, "quality": quality}
        clean_row.pop("raw_response", None)
        clean_rows.append(clean_row)
        text_rows.append(
            {
                "item_id": int(row["item_id"]),
                "title": title,
                "profile_text": build_lora_profile_text(title, cleaned, mode=cfg["profile_text"].get("mode", "lora_full")),
            }
        )
        if max_items > 0 and len(clean_rows) >= max_items:
            break

    clean_path = resolve_project_path(cfg["data"]["profiles_clean"])
    text_path = resolve_project_path(cfg["data"]["profile_text"])
    write_jsonl(clean_path, clean_rows)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with text_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "title", "profile_text"])
        writer.writeheader()
        writer.writerows(text_rows)
    write_json(resolve_project_path(cfg["data"]["quality_report"]), report)
    return [int(row["item_id"]) for row in text_rows], text_rows


def encode_profiles(cfg: dict, text_rows: list[dict]) -> tuple[list[int], "np.ndarray"]:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Profile embedding requires numpy and sentence-transformers.") from exc

    emb_cfg = cfg["embedding"]
    device = resolve_device(str(emb_cfg.get("device", "auto")))
    model = SentenceTransformer(emb_cfg["model_name"], device=device)
    model.max_seq_length = int(emb_cfg.get("max_seq_length", 512))
    item_ids = [int(row["item_id"]) for row in text_rows]
    texts = [row["profile_text"] for row in text_rows]
    embeddings = model.encode(
        texts,
        batch_size=int(emb_cfg["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
    ).astype("float32")

    ids_path = resolve_project_path(cfg["data"]["profile_item_ids"])
    emb_path = resolve_project_path(cfg["data"]["profile_embeddings"])
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, np.asarray(item_ids, dtype="int64"))
    np.save(emb_path, embeddings)
    write_json(
        resolve_project_path(cfg["output"]["embedding_config"]),
        {
            "model_name": emb_cfg["model_name"],
            "device": device,
            "num_items": len(item_ids),
            "embedding_dim": int(embeddings.shape[1]),
            "source": cfg["data"]["profile_text"],
        },
    )
    return item_ids, embeddings


def build_faiss_index(cfg: dict, item_ids: list[int], embeddings: "np.ndarray") -> None:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("FAISS index building requires faiss-cpu and numpy.") from exc

    vectors = np.ascontiguousarray(embeddings.copy().astype("float32"))
    metric = cfg["retrieval"].get("metric", "cosine")
    if metric == "cosine":
        faiss.normalize_L2(vectors)
        base = faiss.IndexFlatIP(vectors.shape[1])
    elif metric == "ip":
        base = faiss.IndexFlatIP(vectors.shape[1])
    else:
        base = faiss.IndexFlatL2(vectors.shape[1])

    index = faiss.IndexIDMap2(base)
    index.add_with_ids(vectors, np.asarray(item_ids, dtype="int64"))
    index_path = resolve_project_path(cfg["data"]["profile_index"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    write_json(
        resolve_project_path(cfg["data"]["index_config"]),
        {"metric": metric, "num_vectors": int(index.ntotal), "embedding_dim": int(vectors.shape[1])},
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    profile_ids, profile_embeddings = encode_profiles(cfg, clean_profiles(cfg, args.max_items)[1])
    build_faiss_index(cfg, profile_ids, profile_embeddings)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or list(cfg["retrieval"]["methods"])
    ks = [int(k) for k in cfg["retrieval"]["ks"]]
    max_eval_users_value = (
        args.max_eval_users
        if args.max_eval_users is not None
        else int(cfg["retrieval"].get("max_eval_users", -1))
    )
    max_eval_users = None if max_eval_users_value == -1 else max_eval_users_value
    item_ids = load_item_ids(resolve_project_path(cfg["data"]["item_ids"]))
    feature_config = load_feature_config(resolve_project_path(cfg["data"]["feature_config"]))
    train_path = resolve_project_path(cfg["data"]["train_samples"])
    eval_path = resolve_project_path(cfg["data"][f"{args.eval_split}_samples"])

    report = {"eval_split": args.eval_split, "methods": {}}
    for method in methods:
        method_item_ids, vectors = build_feature_matrix(
            method=method,
            item_ids=item_ids,
            feature_config=feature_config,
            profile_ids=profile_ids,
            profile_embeddings=profile_embeddings,
            fusion_weights=cfg["retrieval"].get("fusion_weights"),
        )
        metrics = evaluate_content_recall(
            item_vectors=vectors,
            item_ids=method_item_ids,
            train_path=train_path,
            eval_path=eval_path,
            ks=ks,
            max_history_len=int(cfg["retrieval"]["max_history_len"]),
            max_eval_users=max_eval_users,
            user_batch_size=int(cfg["retrieval"].get("user_batch_size", 256)),
        )
        report["methods"][method] = metrics
        print(f"[{method}] {json.dumps(metrics, ensure_ascii=False)}")

    output = resolve_project_path(cfg["output"]["metrics"])
    write_json(output, report)
    print("=" * 80)
    print("V5-LoRA Step 06: Eval LoRA Profiles")
    print("=" * 80)
    print(f"metrics: {project_relative(output)}")


if __name__ == "__main__":
    main()
