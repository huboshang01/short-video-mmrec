"""
V5 Step 06: 评估 Profile 内容向量对召回的增强效果。

评估方法：用户向量=训练历史 item 内容向量均值，在候选库上做 full-catalog ranking。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.eval.content_recall import evaluate_content_recall
from src.v5.profile.io import load_item_ids, load_yaml, write_json
from src.v5.profile.paths import project_relative, resolve_project_path
from src.v5.retrieval.embeddings import build_feature_matrix, load_feature_config, load_profile_embeddings


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5" / "profile_retrieval.yaml"


def parse_args() -> argparse.Namespace:
    """解析评估参数。

    --methods 可只跑某一路召回，例如 profile；默认跑配置里的 title/profile/multimodal/fusion。
    --max-eval-users 适合本地快速冒烟测试，正式结果使用 -1/None 全量评估。
    """
    parser = argparse.ArgumentParser(description="Evaluate V5 profile retrieval with full-catalog ranking.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--eval-split", type=str, default="test", choices=["val", "test"])
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated methods: title,profile,multimodal,fusion.",
    )
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))

    # V3 产物提供 item 顺序、官方多模态特征路径和 train/val/test 行为样本。
    item_ids = load_item_ids(resolve_project_path(cfg["data"]["item_ids"]))
    feature_config = load_feature_config(resolve_project_path(cfg["data"]["feature_config"]))

    # V5 产物提供 MLLM profile 的文本 embedding，用于 profile / fusion 两路实验。
    profile_ids, profile_embeddings = load_profile_embeddings(
        resolve_project_path(cfg["data"]["profile_item_ids"]),
        resolve_project_path(cfg["data"]["profile_embeddings"]),
    )
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or list(cfg["retrieval"]["methods"])
    ks = [int(k) for k in cfg["retrieval"]["ks"]]
    max_eval_users_value = (
        args.max_eval_users
        if args.max_eval_users is not None
        else int(cfg["retrieval"].get("max_eval_users", -1))
    )
    max_eval_users = None if max_eval_users_value == -1 else max_eval_users_value
    eval_path = resolve_project_path(cfg["data"][f"{args.eval_split}_samples"])
    train_path = resolve_project_path(cfg["data"]["train_samples"])
    output = resolve_project_path(args.output or cfg["output"]["metrics"])

    report = {"eval_split": args.eval_split, "methods": {}}
    for method in methods:
        # 将同一批 item 构造成不同内容向量，保证各路召回在同一评估口径下比较。
        method_item_ids, vectors = build_feature_matrix(
            method=method,
            item_ids=item_ids,
            feature_config=feature_config,
            profile_ids=profile_ids,
            profile_embeddings=profile_embeddings,
            fusion_weights=cfg["retrieval"].get("fusion_weights"),
        )
        # 用户向量由 train 历史 item 向量均值得到，再在全量 item 上做 ranking。
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
        print(f"[{method}] {metrics}")

    write_json(output, report)
    print("=" * 80)
    print("V5 Step 06: Eval Profile Recall")
    print("=" * 80)
    print(f"output: {project_relative(output)}")


if __name__ == "__main__":
    main()
