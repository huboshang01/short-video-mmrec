from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.behavior_adapter import count_trainable_parameters
from src.models.item_encoder import ItemEncoder


DEFAULT_EMB_CONFIG = PROJECT_ROOT / "outputs" / "v2" / "embeddings" / "embedding_config.json"


# 命令行参数：控制检查用的 embedding cache、模型维度、设备和 batch 大小
def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check V2 ItemEncoder / BehaviorAdapter.")

    parser.add_argument("--embedding-config", type=str, default=str(DEFAULT_EMB_CONFIG), help="Path to V2 embedding_config.json.")
    parser.add_argument("--embedding-path", type=str, default="", help="Optional override for item_text_embeddings.npy.")
    parser.add_argument("--output-dim", type=int, default=0, help="0 means using embedding_dim from config.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used by the adapter projection head.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of cached item embeddings used for the check.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for the sanity check.")
    parser.add_argument("--use-input-projection", action=argparse.BooleanOptionalAction, default=False, help="Whether to force a Linear input projection.")
    parser.add_argument("--normalize-output", action=argparse.BooleanOptionalAction, default=True, help="Whether ItemEncoder should L2-normalize output embeddings.")

    args = parser.parse_args()
    if args.output_dim < 0:
        raise ValueError("--output-dim cannot be negative.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    return args


# 路径处理：支持绝对路径，也支持相对于项目根目录的配置路径
def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# 设备选择：默认优先使用 CUDA，不可用时回退到 CPU
def get_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device_arg


# 配置读取：加载 Step 3 生成的 embedding_config.json
def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Embedding config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 向量读取：加载 item_text_embeddings.npy，并做基础合法性检查
def load_embeddings(path: Path, expected_dim: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    embeddings = np.load(path).astype("float32")
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
    if embeddings.shape[1] != expected_dim:
        raise ValueError(f"Embedding dim mismatch: got {embeddings.shape[1]}, expected {expected_dim}")
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf.")

    return np.ascontiguousarray(embeddings)


# 模型检查：验证 ItemEncoder 在 train/eval 模式下都能正常前向
def run_forward_check(model: ItemEncoder, x: torch.Tensor, output_dim: int, normalize_output: bool) -> None:
    model.train()
    y_train = model(x)

    model.eval()
    with torch.no_grad():
        y_eval = model(x)
        y_alias = model.encode_item(x)

    print("\n[Forward Check]")
    print(f"input shape:        {tuple(x.shape)}")
    print(f"train output shape: {tuple(y_train.shape)}")
    print(f"eval output shape:  {tuple(y_eval.shape)}")

    if y_eval.shape[-1] != output_dim:
        raise RuntimeError(f"Output dim mismatch: got {y_eval.shape[-1]}, expected {output_dim}")
    if not torch.allclose(y_eval, y_alias):
        raise RuntimeError("encode_item() output is not equal to forward() in eval mode.")

    if normalize_output:
        norms = y_eval.norm(dim=-1).detach().cpu().numpy()
        print("\n[Norm Check]")
        print(f"eval output norm: mean={norms.mean():.6f}, min={norms.min():.6f}, max={norms.max():.6f}")
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise RuntimeError("Normalized output norm is not close to 1.")


def main():
    args = parse_args()

    config_path = resolve_path(args.embedding_config)
    config = load_config(config_path)

    embedding_path = resolve_path(args.embedding_path) if args.embedding_path else resolve_path(config["embedding_path"])
    input_dim = int(config["embedding_dim"])
    output_dim = int(args.output_dim or input_dim)
    device = get_device(args.device)

    print("=" * 80)
    print("V2 Item Adapter Sanity Check")
    print("=" * 80)
    print(f"[INFO] project root: {PROJECT_ROOT}")
    print(f"[INFO] embedding config: {config_path}")
    print(f"[INFO] embeddings: {embedding_path}")
    print(f"[INFO] input_dim: {input_dim}")
    print(f"[INFO] output_dim: {output_dim}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] batch size: {args.batch_size}")

    embeddings = load_embeddings(embedding_path, expected_dim=input_dim)
    batch_size = min(args.batch_size, len(embeddings))
    x = torch.from_numpy(embeddings[:batch_size]).to(device)

    model = ItemEncoder(
        text_dim=input_dim,
        output_dim=output_dim,
        dropout=args.dropout,
        use_input_projection=args.use_input_projection,
        normalize_output=args.normalize_output,
    ).to(device)

    print("\n[Model]")
    print(model)
    print(f"[INFO] trainable parameters: {count_trainable_parameters(model)}")

    run_forward_check(model, x, output_dim=output_dim, normalize_output=args.normalize_output)

    print("\n[Done] Item adapter sanity check passed.")


if __name__ == "__main__":
    main()
