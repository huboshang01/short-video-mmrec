"""轻量 retrieval-aware teacher profile 构造。

第一版不引入行为邻居和 hard negatives，只用 title + V5 clean profile
强化标题关键词、检索词和兴趣标签，避免把 LoRA 训练目标做得过重。
"""

from __future__ import annotations

import re
import ast
from typing import Any


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "in",
    "is",
    "new",
    "of",
    "on",
    "the",
    "this",
    "to",
    "with",
}


def dedupe(values: list[str], max_len: int = 8) -> list[str]:
    cleaned = []
    for value in values:
        text = str(value).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:max_len]


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


def scalar_text(value: Any, fallback: str = "") -> str:
    """把 V5 中偶发的 list 标量字段转成可读文本。"""
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            value = parsed
    if isinstance(value, list):
        text = "，".join(str(x).strip() for x in value if str(x).strip())
    else:
        text = str(value).strip() if value is not None else ""
    return text or fallback


def extract_title_keywords(title: str, max_len: int = 8) -> list[str]:
    """从标题中抽取轻量关键词，优先保留实体词、动作词和数字词。"""
    parts = re.split(r"[\s#，,。:：!！?？/\\|()\[\]{}<>\"']+", title)
    keywords = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        if token.lower() in STOP_WORDS and not token.isdigit():
            continue
        keywords.append(token)
    return dedupe(keywords, max_len=max_len)


def build_retrieval_aware_profile(title: str, base_profile: dict) -> dict:
    """把 V5 clean profile 转成更适合召回的 teacher profile。

    输入：
        title: item 原始标题，作为强召回信号保留。
        base_profile: V5 clean profile 中的 profile 字段。

    输出：
        retrieval-aware profile，供 SFT 和后续 embedding 使用。
    """
    title_keywords = extract_title_keywords(title)
    base_topics = as_list(base_profile.get("main_topics"))
    base_queries = as_list(base_profile.get("search_queries"))
    visual_objects = as_list(base_profile.get("visual_objects"))
    content_type = scalar_text(base_profile.get("content_type"))

    search_queries = dedupe([title, *title_keywords, *base_queries, *base_topics], max_len=8)
    main_topics = dedupe([*title_keywords, *base_topics], max_len=8)
    interest_tags = dedupe([*title_keywords, content_type, *base_topics, *visual_objects[:3]], max_len=8)

    return {
        "title_keywords": title_keywords or [title.strip() or "未知标题"],
        "interest_tags": interest_tags or base_topics or [title.strip() or "未知内容"],
        "summary": str(base_profile.get("summary", title)).strip() or title,
        "main_topics": main_topics or base_topics or [title.strip() or "未知主题"],
        "visual_objects": visual_objects or ["视频画面"],
        "scene": scalar_text(base_profile.get("scene"), "未知"),
        "content_type": content_type or "短视频内容",
        "style": scalar_text(base_profile.get("style"), "未知"),
        "emotion": scalar_text(base_profile.get("emotion"), "未知"),
        "target_audience": scalar_text(base_profile.get("target_audience"), "短视频用户"),
        "search_queries": search_queries or [title.strip() or "未知内容"],
    }
