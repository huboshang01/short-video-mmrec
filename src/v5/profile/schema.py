"""V5 语义 Profile schema、prompt、清洗与文本化。

Profile 字段保持少而稳定：既方便 MLLM 输出 JSON，也方便后续编码成 embedding。
"""

from __future__ import annotations

import json
import re
from typing import Any


PROFILE_FIELDS = [
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
LIST_FIELDS = {"main_topics", "visual_objects", "search_queries"}


def build_profile_prompt(title: str, language: str = "zh") -> str:
    """构造稳定 JSON 输出 prompt，减少后续清洗成本。"""
    if language == "en":
        return (
            "Analyze this short-video item from the frames and title. "
            "Return only one valid JSON object with keys: summary, main_topics, "
            "visual_objects, scene, content_type, style, emotion, target_audience, "
            "search_queries. Use concise English.\n"
            f"Title: {title}"
        )
    return (
        "你是短视频内容理解助手。请根据 5 张视频帧和标题，生成推荐召回可用的语义 Profile。\n"
        "只输出一个合法 JSON，不要输出解释、Markdown 或代码块。字段必须包含：\n"
        "summary, main_topics, visual_objects, scene, content_type, style, emotion, "
        "target_audience, search_queries。\n"
        "要求：summary 用一句话概括内容；list 字段每个保留 3-8 个短词；避免臆造画面中不存在的细节。\n"
        f"标题：{title}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取第一个 JSON object，兼容偶发代码块或前后缀文本。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    decoder = json.JSONDecoder()

    # raw_decode 会正确处理字符串里的花括号，比手写括号计数更稳。
    for start in [0, *[idx for idx, char in enumerate(text) if char == "{"]]:
        try:
            value, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No JSON object found.")


def _clean_list(value: Any, max_len: int = 8) -> list[str]:
    if isinstance(value, str):
        parts = re.split(r"[,，、;/；\n]+", value)
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


def _clean_text(value: Any, fallback: str = "未知") -> str:
    text = str(value).strip() if value is not None else ""
    return re.sub(r"\s+", " ", text) or fallback


def clean_profile(profile: dict[str, Any], title: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    """把任意模型输出归一化为固定 schema，并返回质量标记。"""
    cleaned: dict[str, Any] = {}
    quality = {"missing_fields": [], "empty_list_fields": []}

    for field in PROFILE_FIELDS:
        if field not in profile:
            quality["missing_fields"].append(field)

        # list 字段会进入 embedding 文本，去重和截断能减少模型偶发的重复输出。
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


def mock_profile_from_title(title: str) -> dict[str, Any]:
    """本地无模型时的自检后端：生成可解析 profile，不代表真实 MLLM 效果。"""
    title = title.strip() or "未知短视频"
    words = [w for w in re.split(r"[\s#，,。:：!！?？]+", title) if w][:6]
    return {
        "summary": f"该短视频主要围绕“{title}”展开。",
        "main_topics": words or [title],
        "visual_objects": ["视频画面", "主体人物或物体", "场景元素"],
        "scene": "短视频内容场景",
        "content_type": "泛娱乐或生活内容",
        "style": "标题驱动",
        "emotion": "轻松或好奇",
        "target_audience": "对该主题感兴趣的短视频用户",
        "search_queries": [title, *words[:4]],
    }


def profile_to_text(profile: dict[str, Any]) -> str:
    """将结构化 profile 转成稠密语义文本，作为 embedding 的输入。"""
    lines = [
        f"Summary: {profile['summary']}",
        f"Topics: {', '.join(profile['main_topics'])}",
        f"Visual objects: {', '.join(profile['visual_objects'])}",
        f"Scene: {profile['scene']}",
        f"Content type: {profile['content_type']}",
        f"Style: {profile['style']}",
        f"Emotion: {profile['emotion']}",
        f"Target audience: {profile['target_audience']}",
        f"Search queries: {', '.join(profile['search_queries'])}",
    ]
    return "\n".join(lines)