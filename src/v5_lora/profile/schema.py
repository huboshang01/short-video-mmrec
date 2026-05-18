"""V5-LoRA 推荐检索版 Profile schema 与清洗函数。

V5-LoRA 保留 V5 的解释性字段，同时新增 title_keywords 和 interest_tags，
让 profile 显式承载标题关键词与推荐召回兴趣标签。
"""

from __future__ import annotations

import ast
from typing import Any

PROFILE_FIELDS = [
    "title_keywords",
    "interest_tags",
    "summary",
    "main_topics",
    "visual_objects",
    "scene",
    "content_type",
    "style",
    "emotion",
    "target_audience",
    "search_queries",
]
LIST_FIELDS = {"title_keywords", "interest_tags", "main_topics", "visual_objects", "search_queries"}


def _clean_text(value: Any, fallback: str = "未知") -> str:
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            value = parsed
    if isinstance(value, list):
        value = "，".join(str(x).strip() for x in value if str(x).strip())
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _clean_list(value: Any, max_len: int = 8) -> list[str]:
    """把字符串或列表字段归一化为去重短词列表。"""
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        value = parsed if isinstance(parsed, list) else value
    if isinstance(value, str):
        parts = value.replace("，", ",").replace("、", ",").replace("；", ",").split(",")
    elif isinstance(value, list):
        parts = value
    else:
        parts = []

    cleaned = []
    for part in parts:
        text = str(part).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:max_len]


def clean_lora_profile(profile: dict[str, Any], title: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    """清洗 LoRA profile，并返回质量标记。

    输入：
        profile: 模型输出或规则构造的 profile 字典。
        title: 当前 item 标题，用于必要字段兜底。

    输出：
        cleaned: 固定 V5-LoRA schema 的 profile。
        quality: 缺字段、空列表等轻量质量信息。
    """
    cleaned: dict[str, Any] = {}
    quality = {"missing_fields": [], "empty_list_fields": []}

    for field in PROFILE_FIELDS:
        if field not in profile:
            quality["missing_fields"].append(field)
        if field in LIST_FIELDS:
            values = _clean_list(profile.get(field))
            if not values:
                quality["empty_list_fields"].append(field)
                values = [_clean_text(title, "未知内容")]
            cleaned[field] = values
        else:
            fallback = title if field == "summary" and title else "未知"
            cleaned[field] = _clean_text(profile.get(field), fallback=fallback)

    quality["is_valid"] = not quality["missing_fields"]
    return cleaned, quality
