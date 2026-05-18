"""
V5-LoRA Step 05: 使用 LoRA VLM 批量生成 profile。

默认 backend=mock 仅用于本地链路自检；正式生成时将配置或命令行改成
backend=qwen2_5_vl_lora，并提供 base model 与 adapter 路径。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v5.profile.io import load_yaml, read_jsonl
from src.v5.profile.paths import project_relative, resolve_project_path
from src.v5_lora.profile.prompt import build_lora_profile_prompt
from src.v5_lora.profile.teacher_profile import build_retrieval_aware_profile, extract_title_keywords


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "v5_lora" / "profile_generation_lora.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate V5-LoRA profiles with LoRA VLM.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--backend", type=str, default="", choices=["", "mock", "qwen2_5_vl_lora"])
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--adapter-name-or-path", type=str, default="")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_done_item_ids(output_path: Path) -> set[int]:
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(int(json.loads(line)["item_id"]))
    return done


def build_messages(item: dict) -> list[dict]:
    content = []
    for frame in item["frames"]:
        content.append({"type": "image", "image": f"file://{resolve_project_path(frame)}"})
    content.append({"type": "text", "text": build_lora_profile_prompt(item["title"])})
    return [{"role": "user", "content": content}]


def load_lora_backend(cfg: dict):
    """加载 Qwen2.5-VL base model、processor 和 LoRA adapter。"""
    try:
        import torch
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("LoRA generation requires torch, transformers, peft and qwen-vl-utils.") from exc

    gen_cfg = cfg["generation"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        gen_cfg["model_name"],
        torch_dtype="auto" if gen_cfg.get("torch_dtype", "auto") == "auto" else getattr(torch, gen_cfg["torch_dtype"]),
        device_map=gen_cfg.get("device_map", "auto"),
    )
    adapter_path = gen_cfg.get("adapter_name_or_path", "")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    processor = AutoProcessor.from_pretrained(
        gen_cfg["model_name"],
        min_pixels=int(gen_cfg.get("min_pixels", 200704)),
        max_pixels=int(gen_cfg.get("max_pixels", 802816)),
    )
    model.eval()
    return torch, model, processor, process_vision_info


def generate_one_with_lora(item: dict, cfg: dict, backend) -> dict:
    torch, model, processor, process_vision_info = backend
    gen_cfg = cfg["generation"]
    messages = build_messages(item)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    target_device = getattr(model, "device", next(model.parameters()).device)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(target_device)

    temperature = float(gen_cfg.get("temperature", 0.0))
    generate_kwargs = {"max_new_tokens": int(gen_cfg.get("max_new_tokens", 512))}
    if temperature > 0:
        generate_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": float(gen_cfg.get("top_p", 0.9))})

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids, strict=False)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {**item, "backend": "qwen2_5_vl_lora", "model_name": gen_cfg["model_name"], "raw_response": raw}


def mock_profile(item: dict) -> dict:
    title = item["title"]
    keywords = extract_title_keywords(title)
    base_profile = {
        "summary": title,
        "main_topics": keywords,
        "visual_objects": ["视频画面"],
        "scene": "短视频场景",
        "content_type": "短视频内容",
        "style": "标题驱动",
        "emotion": "中性",
        "target_audience": "对该主题感兴趣的短视频用户",
        "search_queries": [title, *keywords],
    }
    return build_retrieval_aware_profile(title, base_profile)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_project_path(args.config))
    if args.model_name:
        cfg["generation"]["model_name"] = args.model_name
    if args.adapter_name_or_path:
        cfg["generation"]["adapter_name_or_path"] = args.adapter_name_or_path

    input_path = resolve_project_path(args.input or cfg["data"]["profile_inputs"])
    output_path = resolve_project_path(args.output or cfg["data"]["profiles_raw"])
    backend = args.backend or cfg["generation"].get("backend", "mock")
    max_items = args.max_items if args.max_items is not None else int(cfg["generation"].get("max_items", -1))
    items = read_jsonl(input_path)
    if max_items > 0:
        items = items[:max_items]

    done_item_ids = load_done_item_ids(output_path) if args.resume else set()
    if done_item_ids:
        items = [item for item in items if int(item["item_id"]) not in done_item_ids]
    lora_backend = load_lora_backend(cfg) if backend == "qwen2_5_vl_lora" else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume else "w"
    count = 0
    with output_path.open(output_mode, encoding="utf-8") as f:
        for item in items:
            if backend == "mock":
                profile = mock_profile(item)
                row = {**item, "backend": "mock", "raw_response": json.dumps(profile, ensure_ascii=False), "profile": profile}
            else:
                row = generate_one_with_lora(item, cfg, lora_backend)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            count += 1

    print("=" * 80)
    print("V5-LoRA Step 05: Generate LoRA Profiles")
    print("=" * 80)
    print(f"backend: {backend}")
    print(f"output: {project_relative(output_path)}")
    print(f"rows: {count}")


if __name__ == "__main__":
    main()
