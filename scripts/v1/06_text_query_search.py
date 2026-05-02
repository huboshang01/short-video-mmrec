from pathlib import Path
import argparse
import json
import re

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = PROJECT_ROOT / "outputs" / "embeddings"
INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 命令行参数：指定文本 query、返回数量、模型设备，以及索引/元数据/配置文件位置
def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-video semantic search for KuaiRec V1.")

    parser.add_argument("--query", type=str, required=True, help="Text query, e.g. 篮球教学 / 美食探店 / 宠物猫")
    parser.add_argument("--topk", type=int, default=10, help="Number of retrieved videos.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for query encoding.")
    parser.add_argument("--use-query-instruction", action="store_true", help="Add BGE-style query instruction before encoding query.")
    parser.add_argument("--save", action="store_true", help="Save retrieval result to outputs/reports.")
    parser.add_argument("--index-path", type=str, default=str(INDEX_DIR / "video_text_faiss.index"), help="Path to FAISS index.")
    parser.add_argument("--meta", type=str, default=str(EMB_DIR / "video_text_meta.csv"), help="Path to video text metadata.")
    parser.add_argument("--embedding-config", type=str, default=str(EMB_DIR / "embedding_config.json"), help="Path to embedding config.")
    parser.add_argument("--index-config", type=str, default=str(INDEX_DIR / "faiss_index_config.json"), help="Path to FAISS index config.")

    args = parser.parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive.")
    return args


# 设备选择：默认优先使用 CUDA，不可用时回退到 CPU
def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return device_arg


# JSON 读取：加载 embedding 和 FAISS 建索引阶段保存的配置
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 资源加载：读取 FAISS 索引、视频文本元数据、embedding 配置和 index 配置
def load_resources(args):
    index_path = Path(args.index_path)
    meta_path = Path(args.meta)
    embedding_config_path = Path(args.embedding_config)
    index_config_path = Path(args.index_config)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    index = faiss.read_index(str(index_path))

    meta_df = pd.read_csv(
        meta_path,
        encoding="utf-8-sig",
        lineterminator="\n",
        low_memory=False,
    )

    if "video_id" not in meta_df.columns or "video_text" not in meta_df.columns:
        raise ValueError("video_text_meta.csv must contain video_id and video_text columns.")

    embedding_config = load_json(embedding_config_path)
    index_config = load_json(index_config_path)

    if index.ntotal != len(meta_df):
        raise ValueError(f"Mismatch: index.ntotal={index.ntotal}, meta rows={len(meta_df)}")

    embedding_dim = embedding_config.get("embedding_dim")
    if embedding_dim is not None and index.d != int(embedding_dim):
        raise ValueError(f"Mismatch: index dim={index.d}, embedding config dim={embedding_dim}")

    return index, meta_df, embedding_config, index_config


# Query 文本构造：清理空白，并可选加上 BGE 检索指令
def build_query_text(query: str, use_query_instruction: bool) -> str:
    query = query.strip()

    if not query:
        raise ValueError("Query is empty.")

    if use_query_instruction:
        # BGE 中文检索常用 query instruction。
        # V1 默认不开启，避免和已编码的视频侧文本分布差异过大。
        return f"为这个句子生成表示以用于检索相关文章：{query}"

    return query


# 模型加载：使用和 item 文本编码阶段一致的 SentenceTransformer 模型
def load_model(model_name: str, device: str, max_seq_length: int) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    return model


# Query 编码：将用户输入文本转成向量，归一化策略要和索引构建方式保持一致
def encode_query(model: SentenceTransformer, query_text: str, normalize: bool) -> np.ndarray:
    query_embedding = model.encode(
        [query_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    query_embedding = query_embedding.astype("float32")
    query_embedding = np.ascontiguousarray(query_embedding)

    if np.isnan(query_embedding).any() or np.isinf(query_embedding).any():
        raise ValueError("Query embedding contains NaN or Inf.")

    return query_embedding


# 文本元数据映射：建立 video_id 到 video_text 的查找表
def build_text_lookup(meta_df: pd.DataFrame) -> dict[int, str]:
    meta_df = meta_df.copy()
    meta_df["video_id"] = pd.to_numeric(meta_df["video_id"], errors="raise").astype("int64")
    meta_df["video_text"] = meta_df["video_text"].fillna("").astype(str)
    meta_df = meta_df.drop_duplicates("video_id")
    return dict(zip(meta_df["video_id"], meta_df["video_text"]))


# FAISS 检索：用 query 向量搜索相似视频，并拼回可读的 video_text。注意这里区别于05，并没有“过滤自己”，因为用户输入的 query 不一定和某个视频文本完全一样。
def search(index, query_embedding: np.ndarray, id_to_text: dict[int, str], topk: int) -> pd.DataFrame:
    query_vector = query_embedding.copy().astype("float32")
    query_vector = np.ascontiguousarray(query_vector)

    search_k = min(topk, index.ntotal)
    scores, video_ids = index.search(query_vector, search_k)

    results = []
    for score, vid in zip(scores[0], video_ids[0]):
        vid = int(vid)

        if vid == -1:
            continue

        results.append(
            {
                "rank": len(results) + 1,
                "video_id": vid,
                "score": float(score),
                "video_text": id_to_text.get(vid, ""),
            }
        )

    return pd.DataFrame(results)


# 文件名清理：把 query 转成适合保存 CSV 的短文件名
def safe_filename(text: str, max_len: int = 30) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\u4e00-\u9fff-]", "", text)
    if not text:
        text = "query"
    return text[:max_len]


# 结果展示：在终端中打印原始 query、实际编码 query 和召回视频
def print_results(query: str, query_text_for_encoding: str, result_df: pd.DataFrame):
    print("=" * 100)
    print("[TEXT QUERY]")
    print("raw query:", query)
    print("encoded query:", query_text_for_encoding)

    print("=" * 100)
    print("[RETRIEVED VIDEOS]")

    if result_df.empty:
        print("No results found.")
        return

    for _, row in result_df.iterrows():
        print("-" * 100)
        print(f"rank: {int(row['rank'])}")
        print(f"video_id: {int(row['video_id'])}")
        print(f"score: {row['score']:.6f}")
        print("video_text:")
        print(str(row["video_text"])[:800])


# 结果保存：可选地把文本检索结果保存为 CSV
def save_results(query: str, topk: int, result_df: pd.DataFrame) -> Path:
    filename = safe_filename(query)
    out_path = REPORT_DIR / f"text_query_{filename}_top{topk}.csv"

    save_df = result_df.copy()
    save_df.insert(0, "query", query)
    save_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main():
    args = parse_args()

    print("=" * 100)
    print("Text Query Search - KuaiRec V1")
    print("=" * 100)
    print(f"[INFO] query: {args.query}")
    print(f"[INFO] topk: {args.topk}")

    index, meta_df, embedding_config, index_config = load_resources(args)

    model_name = embedding_config.get("model_name")
    max_seq_length = int(embedding_config.get("max_seq_length", 256))
    metric = index_config.get("metric", "cosine")
    device = get_device(args.device)

    if not model_name:
        raise ValueError("model_name not found in embedding_config.json")
    if metric not in {"cosine", "ip", "l2"}:
        raise ValueError(f"Unsupported index metric: {metric}")

    query_normalize = bool(embedding_config.get("normalize", False)) or metric == "cosine"

    print(f"[INFO] model: {model_name}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] max_seq_length: {max_seq_length}")
    print(f"[INFO] metric: {metric}")
    print(f"[INFO] query normalize: {query_normalize}")
    print(f"[INFO] index.ntotal: {index.ntotal}")

    query_text_for_encoding = build_query_text(args.query, args.use_query_instruction)
    model = load_model(model_name, device, max_seq_length)

    query_embedding = encode_query(
        model=model,
        query_text=query_text_for_encoding,
        normalize=query_normalize,
    )

    id_to_text = build_text_lookup(meta_df)
    result_df = search(
        index=index,
        query_embedding=query_embedding,
        id_to_text=id_to_text,
        topk=args.topk,
    )

    print_results(args.query, query_text_for_encoding, result_df)

    if args.save:
        out_path = save_results(args.query, args.topk, result_df)
        print("=" * 100)
        print(f"[SAVED] {out_path}")

    print("=" * 100)
    print("Text query search done.")


if __name__ == "__main__":
    main()
