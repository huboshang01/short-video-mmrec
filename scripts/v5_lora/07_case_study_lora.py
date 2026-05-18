"""
V5-LoRA Step 07: 生成 LoRA profile 召回案例分析。

默认做 item-to-item 相似召回；传入 --query 时额外做 query-to-item 检索。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.profile.io import load_item_meta, load_yaml, read_jsonl, write_json
from src.v5.profile.paths import project_relative, resolve_project_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5_lora" / "profile_retrieval_lora.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V5-LoRA profile retrieval case study.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-items", type=int, default=20)
    parser.add_argument("--item-ids", type=str, default="")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Case study requires faiss-cpu and numpy.") from exc

    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    item_meta = {int(row["item_id"]): row for row in load_item_meta(resolve_project_path(cfg["data"]["item_meta"]))}
    profiles = {int(row["item_id"]): row["profile"] for row in read_jsonl(resolve_project_path(cfg["data"]["profiles_clean"]))}
    item_ids = np.load(resolve_project_path(cfg["data"]["profile_item_ids"])).astype("int64")
    embeddings = np.load(resolve_project_path(cfg["data"]["profile_embeddings"])).astype("float32")
    index = faiss.read_index(str(resolve_project_path(cfg["data"]["profile_index"])))
    id_to_row = {int(item_id): idx for idx, item_id in enumerate(item_ids.tolist())}
    metric = str(cfg["retrieval"].get("metric", "cosine"))

    selected = [int(x) for x in args.item_ids.split(",") if x.strip()] if args.item_ids else item_ids[: args.num_items].tolist()
    cases = []
    for item_id in selected:
        if int(item_id) not in id_to_row:
            continue
        query_vec = embeddings[id_to_row[int(item_id)] : id_to_row[int(item_id)] + 1].copy()
        if metric == "cosine":
            faiss.normalize_L2(query_vec)
        scores, neighbors = index.search(query_vec, args.topk + 1)
        retrieved = []
        for score, neighbor_id in zip(scores[0], neighbors[0], strict=False):
            neighbor_id = int(neighbor_id)
            if neighbor_id == int(item_id) or neighbor_id == -1:
                continue
            retrieved.append({"item_id": neighbor_id, "score": float(score), "title": item_meta.get(neighbor_id, {}).get("title", "")})
            if len(retrieved) >= args.topk:
                break
        cases.append(
            {
                "item_id": int(item_id),
                "title": item_meta.get(int(item_id), {}).get("title", ""),
                "profile": profiles.get(int(item_id), {}),
                "similar_items": retrieved,
            }
        )

    report = {"item_to_item_cases": cases}
    if args.query:
        from sentence_transformers import SentenceTransformer

        emb_cfg = cfg["embedding"]
        model = SentenceTransformer(emb_cfg["model_name"], device="cpu")
        query_vec = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=(metric == "cosine")).astype("float32")
        scores, ids = index.search(query_vec, args.topk)
        report["query_to_item"] = {
            "query": args.query,
            "results": [
                {"item_id": int(item_id), "score": float(score), "title": item_meta.get(int(item_id), {}).get("title", "")}
                for score, item_id in zip(scores[0], ids[0], strict=False)
                if int(item_id) != -1
            ],
        }

    output = resolve_project_path(args.output or cfg["output"]["case_study"])
    write_json(output, report)
    print("=" * 80)
    print("V5-LoRA Step 07: Case Study")
    print("=" * 80)
    print(f"output: {project_relative(output)}")
    print(f"cases: {len(cases)}")


if __name__ == "__main__":
    main()
