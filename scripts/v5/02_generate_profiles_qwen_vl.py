"""
V5 Step 02: 使用 MLLM 批量生成短视频语义 Profile。

输入：
    data/processed/v5/microlens_100k/profile_inputs.jsonl
    每行包含 item_id、title、frames、likes、views、category_id 等字段。
    其中只有 frames + title 会进入 MLLM prompt，likes/views/category_id 仅随结果保留为元数据。

输出：
    data/processed/v5/microlens_100k/profiles_raw.jsonl
    每行保留原始 item 字段，并新增 backend、model_name/raw_response 或 mock profile。

默认 backend=mock 只用于本地链路自检；正式生成时在配置里改为 qwen2_5_vl。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.profile.io import read_jsonl, load_yaml
from src.v5.profile.paths import project_relative, resolve_project_path
from src.v5.profile.schema import build_profile_prompt, mock_profile_from_title


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5" / "profile_generation.yaml"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    关键参数：
        --backend mock：不加载大模型，用标题生成假 profile，检查工程链路。
        --backend qwen2_5_vl：加载 Qwen2.5-VL 做真实多图理解。
        --model-name：覆盖配置中的模型名，可传 AutoDL 上的本地权重目录。
        --resume：追加写入输出文件，并跳过已生成 item_id。
    """
    parser = argparse.ArgumentParser(description="Generate V5 semantic profiles with Qwen2.5-VL or mock backend.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--backend", type=str, default="", choices=["", "mock", "qwen2_5_vl"])
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Override generation.model_name; can be a local model path.",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Append to output and skip item_ids already generated.")
    return parser.parse_args()


def build_messages(item: dict, language: str) -> list[dict]:
    """把一条 profile input 转成 Qwen2.5-VL chat messages。

    输入：
        item: 01_build_profile_inputs.py 生成的一行 JSON。
        language: prompt 语言，目前支持 zh / en。

    输出：
        Qwen2.5-VL processor.apply_chat_template 可接收的 messages。

    注意：
        item 中的 likes/views/category_id 不进入 prompt，避免热度和行为信号污染内容语义 Profile。
    """
    content = []
    for frame in item["frames"]:
        content.append({"type": "image", "image": f"file://{resolve_project_path(frame)}"})
    content.append(
        {
            "type": "text",
            "text": build_profile_prompt(item["title"], language),
        }
    )
    return [{"role": "user", "content": content}]


def load_qwen_backend(cfg: dict):
    """加载 Qwen2.5-VL 模型、processor 和视觉输入处理函数。

    输入：
        cfg: profile_generation.yaml 配置字典，使用 generation.model_name、
             device_map、torch_dtype、min_pixels、max_pixels 等字段。

    输出：
        (torch, model, processor, process_vision_info)

    说明：
        该函数只在 backend=qwen2_5_vl 时调用；mock 模式不会下载或加载模型。
    """
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError("Qwen2.5-VL generation requires transformers, torch and qwen-vl-utils.") from exc

    gen_cfg = cfg["generation"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        gen_cfg["model_name"],
        torch_dtype="auto" if gen_cfg.get("torch_dtype", "auto") == "auto" else getattr(torch, gen_cfg["torch_dtype"]),
        device_map=gen_cfg.get("device_map", "auto"),
    )
    processor = AutoProcessor.from_pretrained(
        gen_cfg["model_name"],
        min_pixels=int(gen_cfg.get("min_pixels", 200704)),
        max_pixels=int(gen_cfg.get("max_pixels", 802816)),
    )
    model.eval()
    return torch, model, processor, process_vision_info


def load_done_item_ids(output_path: Path) -> set[int]:
    """读取已生成 item_id，支持云端全量任务断点续跑。

    输入：
        output_path: profiles_raw.jsonl 路径。

    输出：
        已成功写入输出文件的 item_id 集合。
    """
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["item_id"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return done


def generate_one_with_qwen(item: dict, cfg: dict, backend) -> dict:
    """对单个 item 调用 Qwen2.5-VL 生成原始 Profile 文本。

    输入：
        item: profile_inputs.jsonl 中的一行。
        cfg: profile_generation.yaml 配置。
        backend: load_qwen_backend 返回的模型相关对象。

    输出：
        一行 profiles_raw.jsonl 记录，包含原始 item 字段和 raw_response。

    说明：
        raw_response 仍是模型原始字符串，JSON 解析和 schema 校验放在 Step 03。
    """
    torch, model, processor, process_vision_info = backend
    gen_cfg = cfg["generation"]
    messages = build_messages(item, gen_cfg.get("language", "zh"))
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    temperature = float(gen_cfg.get("temperature", 0.0))
    generate_kwargs = {"max_new_tokens": int(gen_cfg.get("max_new_tokens", 512))}
    if temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": float(gen_cfg.get("top_p", 0.9)),
            }
        )

    # 推理阶段不需要梯度，逐条生成后立即写盘，避免全量任务中途失败丢结果。
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids, strict=False)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {**item, "backend": "qwen2_5_vl", "model_name": gen_cfg["model_name"], "raw_response": raw}


def main() -> None:
    """主流程：读取输入、选择后端、逐条生成并写出 JSONL。"""
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    input_path = resolve_project_path(args.input or cfg["data"]["profile_inputs"])
    output_path = resolve_project_path(args.output or cfg["data"]["profiles_raw"])
    backend = args.backend or cfg["generation"].get("backend", "mock")
    if args.model_name:
        cfg["generation"]["model_name"] = args.model_name
    max_items = args.max_items if args.max_items is not None else int(cfg["generation"].get("max_items", -1))
    items = read_jsonl(input_path)
    if max_items and max_items > 0:
        items = items[:max_items]

    done_item_ids = load_done_item_ids(output_path) if args.resume else set()
    if done_item_ids:
        # 云端全量生成可能被抢占或手动中断；resume 时跳过已完成 item。
        items = [item for item in items if int(item["item_id"]) not in done_item_ids]

    qwen_backend = load_qwen_backend(cfg) if backend == "qwen2_5_vl" else None
    if backend not in {"mock", "qwen2_5_vl"}:
        raise ValueError(f"Unsupported backend: {backend}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    output_mode = "a" if args.resume else "w"
    with output_path.open(output_mode, encoding="utf-8") as f:
        for item in items:
            if backend == "mock":
                # mock 只验证下游 JSONL、清洗、embedding 和评估流程，不代表真实 MLLM 质量。
                profile = mock_profile_from_title(item["title"])
                row = {
                    **item,
                    "backend": "mock",
                    "raw_response": json.dumps(profile, ensure_ascii=False),
                    "profile": profile,
                }
            else:
                row = generate_one_with_qwen(item, cfg, qwen_backend)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            count += 1

    print("=" * 80)
    print("V5 Step 02: Generate Profiles")
    print("=" * 80)
    print(f"backend: {backend}")
    print(f"input: {project_relative(input_path)}")
    print(f"output: {project_relative(output_path)}")
    print(f"model: {cfg['generation'].get('model_name')}")
    print(f"skipped by resume: {len(done_item_ids)}")
    print(f"rows: {count}")


if __name__ == "__main__":
    main()
