from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.behavior_adapter import count_trainable_parameters
from src.models.item_encoder import ItemEncoder
from src.models.user_encoder import UserEncoder, UserTower, WatchRatioWeightedPooling


DEFAULT_EMB_CONFIG = PROJECT_ROOT / "outputs" / "v2" / "embeddings" / "embedding_config.json"
DEFAULT_BEHAVIOR_SAMPLES = PROJECT_ROOT / "data" / "processed" / "v2" / "behavior_samples_train.csv"


# 命令行参数：控制检查用的 embedding cache、行为样本、历史长度和设备
def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check V2 UserEncoder / UserTower.")

    parser.add_argument("--embedding-config", type=str, default=str(DEFAULT_EMB_CONFIG), help="Path to V2 embedding_config.json.")
    parser.add_argument("--behavior-samples", type=str, default=str(DEFAULT_BEHAVIOR_SAMPLES), help="Path to behavior_samples_train.csv.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of users used for sanity check.")
    parser.add_argument("--max-history-len", type=int, default=20, help="Maximum history length per user.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used by item/user projection heads.")
    parser.add_argument("--weight-cap", type=float, default=5.0, help="Cap watch_ratio weights before pooling.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for sanity check.")
    parser.add_argument("--use-input-projection", action=argparse.BooleanOptionalAction, default=False, help="Whether to force Linear input projections.")
    parser.add_argument("--normalize-output", action=argparse.BooleanOptionalAction, default=True, help="Whether item/user outputs should be L2-normalized.")

    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_history_len <= 0:
        raise ValueError("--max-history-len must be positive.")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1).")
    if args.weight_cap <= 0:
        raise ValueError("--weight-cap must be positive.")
    return args


# 路径处理：支持绝对路径，也支持相对于项目根目录的路径
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


# 向量读取：加载 item_text_embeddings.npy 和 item_ids.npy
def load_embedding_cache(config: dict) -> tuple[np.ndarray, np.ndarray]:
    embedding_path = resolve_path(config["embedding_path"])
    item_ids_path = resolve_path(config["item_ids_path"])

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    if not item_ids_path.exists():
        raise FileNotFoundError(f"Item ids file not found: {item_ids_path}")

    embeddings = np.load(embedding_path).astype("float32")
    item_ids = np.load(item_ids_path).astype("int64")

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
    if len(item_ids) != embeddings.shape[0]:
        raise ValueError("item_ids and embeddings row count mismatch.")
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf.")

    return np.ascontiguousarray(embeddings), item_ids


# 行为样本读取：只读取 sanity check 需要的三列
def load_behavior_samples(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Behavior samples not found: {path}")

    samples = pd.read_csv(
        path,
        encoding="utf-8-sig",
        lineterminator="\n",
        low_memory=False,
        usecols=["user_id", "item_id", "watch_ratio"],
    )
    samples["user_id"] = pd.to_numeric(samples["user_id"], errors="raise").astype("int64")
    samples["item_id"] = pd.to_numeric(samples["item_id"], errors="raise").astype("int64")
    samples["watch_ratio"] = pd.to_numeric(samples["watch_ratio"], errors="coerce").fillna(0.0).astype("float32")
    return samples


# 样本构造：从行为样本中抽一小批用户历史，转成模型输入需要的 index/weight/mask
def build_user_history_batch(samples: pd.DataFrame, item_id_to_index: dict[int, int], batch_size: int, max_history_len: int):
    samples = samples[samples["item_id"].isin(item_id_to_index.keys())].copy()

    user_groups = []
    for user_id, group in samples.groupby("user_id", sort=False):
        if len(group) >= 2:
            user_groups.append((int(user_id), group.head(max_history_len)))
        if len(user_groups) >= batch_size:
            break

    if not user_groups:
        raise ValueError("No users with at least 2 valid interactions found.")

    history_indices = np.zeros((len(user_groups), max_history_len), dtype=np.int64)
    watch_ratios = np.zeros((len(user_groups), max_history_len), dtype=np.float32)
    mask = np.zeros((len(user_groups), max_history_len), dtype=np.float32)
    user_ids = []

    for row_idx, (user_id, group) in enumerate(user_groups):
        user_ids.append(user_id)
        for col_idx, row in enumerate(group.itertuples(index=False)):
            history_indices[row_idx, col_idx] = item_id_to_index[int(row.item_id)]
            watch_ratios[row_idx, col_idx] = float(row.watch_ratio)
            mask[row_idx, col_idx] = 1.0

    return user_ids, history_indices, watch_ratios, mask


# mask 检查：padding 位置的向量变化不应影响池化结果
def run_mask_check(pooling: WatchRatioWeightedPooling, history_item_embs: torch.Tensor, watch_ratios: torch.Tensor, mask: torch.Tensor) -> None:
    if not (mask == 0).any():
        print("[WARN] No padding positions in this batch, skip mask invariance check.")
        return

    changed_history = history_item_embs.clone()
    changed_history[mask == 0] = torch.randn_like(changed_history[mask == 0]) * 100.0

    pooled_a = pooling(history_item_embs, watch_ratios, mask)
    pooled_b = pooling(changed_history, watch_ratios, mask)
    if not torch.allclose(pooled_a, pooled_b, atol=1e-5):
        raise RuntimeError("Mask check failed: padding positions affected pooled user embedding.")


# 前向检查：验证 UserTower 从历史 BGE 向量到 user embedding 的完整路径
def run_forward_check(user_tower: UserTower, history_text_embs: torch.Tensor, watch_ratios: torch.Tensor, mask: torch.Tensor, normalize_output: bool) -> torch.Tensor:
    user_tower.train()
    user_emb_train = user_tower(history_text_embs, watch_ratios, mask)

    user_tower.eval()
    with torch.no_grad():
        user_emb_eval = user_tower(history_text_embs, watch_ratios, mask)
        user_emb_alias = user_tower.encode_user(history_text_embs, watch_ratios, mask)

    print("\n[Forward Check]")
    print(f"history_text_embs shape: {tuple(history_text_embs.shape)}")
    print(f"watch_ratios shape:      {tuple(watch_ratios.shape)}")
    print(f"mask shape:              {tuple(mask.shape)}")
    print(f"user train shape:        {tuple(user_emb_train.shape)}")
    print(f"user eval shape:         {tuple(user_emb_eval.shape)}")

    if not torch.allclose(user_emb_eval, user_emb_alias):
        raise RuntimeError("UserTower encode_user() output is not equal to forward() in eval mode.")

    if normalize_output:
        norms = user_emb_eval.norm(dim=-1).detach().cpu().numpy()
        print("\n[Norm Check]")
        print(f"eval user embedding norm: mean={norms.mean():.6f}, min={norms.min():.6f}, max={norms.max():.6f}")
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise RuntimeError("Normalized user embedding norm is not close to 1.")

    return user_emb_eval


def main():
    args = parse_args()

    config_path = resolve_path(args.embedding_config)
    samples_path = resolve_path(args.behavior_samples)
    config = load_config(config_path)
    device = get_device(args.device)

    input_dim = int(config["embedding_dim"])
    output_dim = input_dim

    print("=" * 80)
    print("V2 User Encoder Sanity Check")
    print("=" * 80)
    print(f"[INFO] project root: {PROJECT_ROOT}")
    print(f"[INFO] embedding config: {config_path}")
    print(f"[INFO] behavior samples: {samples_path}")
    print(f"[INFO] input_dim/output_dim: {input_dim}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] batch size: {args.batch_size}")
    print(f"[INFO] max history len: {args.max_history_len}")

    item_text_embeddings, item_ids = load_embedding_cache(config)
    samples = load_behavior_samples(samples_path)
    item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(item_ids.tolist())}

    user_ids, history_indices, watch_ratio_values, mask_values = build_user_history_batch(
        samples=samples,
        item_id_to_index=item_id_to_index,
        batch_size=args.batch_size,
        max_history_len=args.max_history_len,
    )

    embedding_table = torch.from_numpy(item_text_embeddings).to(device)
    history_indices = torch.from_numpy(history_indices).to(device)
    watch_ratios = torch.from_numpy(watch_ratio_values).to(device)
    mask = torch.from_numpy(mask_values).to(device)
    history_text_embs = embedding_table[history_indices]

    item_encoder = ItemEncoder(
        text_dim=input_dim,
        output_dim=output_dim,
        dropout=args.dropout,
        use_input_projection=args.use_input_projection,
        normalize_output=args.normalize_output,
    ).to(device)
    user_encoder = UserEncoder(
        input_dim=output_dim,
        output_dim=output_dim,
        dropout=args.dropout,
        use_input_projection=args.use_input_projection,
        normalize_output=args.normalize_output,
        weight_cap=args.weight_cap,
    ).to(device)
    user_tower = UserTower(item_encoder=item_encoder, user_encoder=user_encoder).to(device)

    print("\n[Model]")
    print(user_tower)
    print("\n[Parameters]")
    print(f"item encoder trainable params: {count_trainable_parameters(item_encoder)}")
    print(f"user encoder trainable params: {count_trainable_parameters(user_encoder)}")
    print(f"total trainable params: {count_trainable_parameters(user_tower)}")

    print("\n[Batch]")
    print(f"user_ids: {user_ids}")
    print(f"history_indices shape: {tuple(history_indices.shape)}")
    print(f"valid history counts: {mask.sum(dim=1).detach().cpu().numpy().astype(int).tolist()}")

    user_emb_eval = run_forward_check(user_tower, history_text_embs, watch_ratios, mask, args.normalize_output)

    with torch.no_grad():
        flat_history = history_text_embs.reshape(-1, input_dim)
        flat_item_embs = item_encoder.encode_item(flat_history)
        history_item_embs = flat_item_embs.reshape(history_text_embs.shape[0], history_text_embs.shape[1], -1)
        run_mask_check(user_encoder.pooling, history_item_embs, watch_ratios, mask)

    if user_emb_eval.shape[-1] != output_dim:
        raise RuntimeError(f"User embedding dim mismatch: got {user_emb_eval.shape[-1]}, expected {output_dim}")

    print("\n[Done] User encoder sanity check passed.")


if __name__ == "__main__":
    main()
