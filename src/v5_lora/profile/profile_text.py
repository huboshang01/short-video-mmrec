"""V5-LoRA profile_text 模板。

这里集中管理 Stage 0 消融模板和 LoRA profile 默认文本化模板，
保证 V5 raw profile 与 V5-LoRA profile 可以在同一口径下评估。
"""

from __future__ import annotations

import ast
from typing import Any


ABLATION_METHODS = [
    "title_only",
    "title_queries_topics",
    "title_queries_topics_summary",
    "full_reordered",
]


def as_list(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            value = parsed
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def join_values(value: Any) -> str:
    values = as_list(value)
    return "，".join(values) if values else "未知"


def build_ablation_text(method: str, title: str, profile: dict) -> str:
    """构造 A-D 四组 profile_text 消融文本。"""
    title = title.strip() or "未知标题"
    lines = [f"标题：{title}"]
    if method == "title_only":
        return "\n".join(lines)

    lines.extend(
        [
            f"推荐检索词：{join_values(profile.get('search_queries'))}",
            f"主题：{join_values(profile.get('main_topics'))}",
        ]
    )
    if method == "title_queries_topics":
        return "\n".join(lines)

    lines.append(f"摘要：{profile.get('summary', '未知')}")
    if method == "title_queries_topics_summary":
        return "\n".join(lines)

    if method == "full_reordered":
        lines = [
            f"标题：{title}",
            f"推荐检索词：{join_values(profile.get('search_queries'))}",
            f"主题：{join_values(profile.get('main_topics'))}",
            f"内容类型：{profile.get('content_type', '未知')}",
            f"摘要：{profile.get('summary', '未知')}",
            f"画面元素：{join_values(profile.get('visual_objects'))}",
            f"场景：{profile.get('scene', '未知')}",
            f"风格：{profile.get('style', '未知')}",
            f"情绪：{profile.get('emotion', '未知')}",
            f"受众：{profile.get('target_audience', '未知')}",
        ]
        return "\n".join(lines)
    raise ValueError(f"Unsupported ablation method: {method}")


def build_lora_profile_text(title: str, profile: dict, mode: str = "lora_compact") -> str:
    """构造 V5-LoRA 默认 profile_text。

    输入：
        title: item 原始标题。
        profile: retrieval-aware teacher 或 LoRA 模型生成的 profile。
        mode: lora_compact 用于召回评估，lora_full 用于人工查看完整字段。
    """
    if mode not in {"lora_compact", "lora_full"}:
        raise ValueError(f"Unsupported lora profile text mode: {mode}")
    title = title.strip() or "未知标题"
    lines = [
        f"标题：{title}",
        f"标题关键词：{join_values(profile.get('title_keywords'))}",
        f"推荐检索词：{join_values(profile.get('search_queries'))}",
        f"兴趣标签：{join_values(profile.get('interest_tags'))}",
        f"主题：{join_values(profile.get('main_topics'))}",
    ]
    if mode == "lora_compact":
        return "\n".join(lines)

    lines.extend(
        [
        f"内容类型：{profile.get('content_type', '未知')}",
        f"摘要：{profile.get('summary', '未知')}",
        f"画面元素：{join_values(profile.get('visual_objects'))}",
        f"场景：{profile.get('scene', '未知')}",
        f"风格：{profile.get('style', '未知')}",
        f"情绪：{profile.get('emotion', '未知')}",
        f"受众：{profile.get('target_audience', '未知')}",
        ]
    )
    return "\n".join(lines)
