"""V5-LoRA 多模态 profile 生成 prompt。"""

from __future__ import annotations


def build_lora_profile_prompt(title: str) -> str:
    """构造 LoRA 推理和 SFT 数据使用的用户指令。"""
    return (
        "请根据 5 张短视频帧和标题，生成推荐召回友好的语义 Profile。\n"
        "只输出一个合法 JSON，不要输出解释、Markdown 或代码块。\n"
        "字段必须包含：title_keywords, interest_tags, summary, main_topics, "
        "visual_objects, scene, content_type, style, emotion, target_audience, search_queries。\n"
        "要求：保留标题中的核心实体词、动作词和主题词；search_queries 写成适合召回的短查询；"
        "list 字段每个保留 3-8 个短词；不要臆造画面中不存在的细节。\n"
        f"标题：{title}"
    )
