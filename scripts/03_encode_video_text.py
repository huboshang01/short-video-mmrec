from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMB_DIR = PROJECT_ROOT / "outputs" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)


# 命令行参数：控制输入文件、模型、设备和编码批大小
def parse_args():
    parser = argparse.ArgumentParser(description="Encode KuaiRec video_text into embeddings.")

    parser.add_argument("--input", type=str, default=str(PROCESSED_DIR / "video_text.csv"), help="Path to processed video_text.csv")
    parser.add_argument("--model-name", type=str, default="BAAI/bge-small-zh-v1.5", help="SentenceTransformer model name or local path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for encoding.")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Max sequence length for text encoder.")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True, help="Normalize embeddings for cosine similarity / inner product search.")

    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be positive.")
    return args


# 路径展示：项目内路径保存为相对路径，项目外路径保留绝对路径
def format_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# 设备选择：默认优先使用 CUDA，不可用时回退到 CPU
def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device_arg


# 输入读取：只保留向量编码必须的 video_id 和 video_text，并保证 video_id 唯一有序
def read_video_text(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        lineterminator="\n",
        low_memory=False,
    )

    required_cols = {"video_id", "video_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[["video_id", "video_text"]].copy()
    df["video_id"] = pd.to_numeric(df["video_id"], errors="raise").astype("int64")
    df["video_text"] = df["video_text"].fillna("").astype(str).str.strip()
    df.loc[df["video_text"] == "", "video_text"] = "无文本信息"

    before = len(df)
    df = df.drop_duplicates("video_id").copy()
    after = len(df)
    if before != after:
        print(f"[WARN] drop duplicated video_id rows: {before} -> {after}")

    df = df.sort_values("video_id").reset_index(drop=True)
    return df


# 模型加载：使用 SentenceTransformer 封装好的 BGE 编码器
def load_model(model_name: str, device: str, max_seq_length: int) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    return model


# 文本编码：将每条 video_text 转成一个稠密向量
def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.astype("float32")


# 结果校验：确认 video_id 与向量行数一一对应，且向量中没有非法数值
def check_embeddings(video_ids: np.ndarray, embeddings: np.ndarray, normalize: bool) -> None:
    print("=" * 80)
    print("[CHECK]")
    print("video_ids shape:", video_ids.shape)
    print("embeddings shape:", embeddings.shape)
    print("embedding dtype:", embeddings.dtype)
    print("has nan:", np.isnan(embeddings).any())
    print("has inf:", np.isinf(embeddings).any())

    if len(video_ids) != embeddings.shape[0]:
        raise ValueError("video_ids and embeddings row count mismatch.")

    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf.")

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1)
        print("norm min:", norms.min())
        print("norm max:", norms.max())
        print("norm mean:", norms.mean())


# 输出保存：保存向量矩阵、video_id 映射、元数据和本次编码配置
def save_outputs(
    df: pd.DataFrame,
    video_ids: np.ndarray,
    embeddings: np.ndarray,
    args,
    device: str,
) -> None:
    emb_path = EMB_DIR / "video_text_embeddings.npy"
    ids_path = EMB_DIR / "video_ids.npy"
    meta_path = EMB_DIR / "video_text_meta.csv"
    config_path = EMB_DIR / "embedding_config.json"

    np.save(emb_path, embeddings)
    np.save(ids_path, video_ids)
    df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    config = {
        "model_name": args.model_name,
        "input_path": format_path(Path(args.input)),
        "device": device,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "normalize": args.normalize,
        "num_videos": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_path": format_path(emb_path),
        "video_ids_path": format_path(ids_path),
        "meta_path": format_path(meta_path),
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[SAVED]")
    print(f"embeddings: {emb_path}")
    print(f"video ids: {ids_path}")
    print(f"meta: {meta_path}")
    print(f"config: {config_path}")


def main():
    args = parse_args()

    input_path = Path(args.input)
    device = get_device(args.device)

    print("=" * 80)
    print("Encode KuaiRec video_text")
    print("=" * 80)
    print(f"[INFO] project root: {PROJECT_ROOT}")
    print(f"[INFO] input: {input_path}")
    print(f"[INFO] model: {args.model_name}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] batch size: {args.batch_size}")
    print(f"[INFO] max seq length: {args.max_seq_length}")
    print(f"[INFO] normalize: {args.normalize}")

    df = read_video_text(input_path)

    video_ids = df["video_id"].to_numpy()
    texts = df["video_text"].tolist()

    print("=" * 80)
    print("[DATA]")
    print(f"num videos: {len(df)}")
    print("sample:")
    for i in range(min(3, len(df))):
        print("-" * 80)
        print("video_id:", video_ids[i])
        print("video_text:", texts[i][:300])

    print("=" * 80)
    print("[MODEL LOADING]")
    model = load_model(args.model_name, device, args.max_seq_length)

    print("[ENCODING]")
    embeddings = encode_texts(model, texts, args.batch_size, args.normalize)

    check_embeddings(video_ids, embeddings, args.normalize)
    save_outputs(df, video_ids, embeddings, args, device)

    print("=" * 80)
    print("Encode video_text done.")


if __name__ == "__main__":
    main()
