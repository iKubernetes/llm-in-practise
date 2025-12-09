import uuid
import os
from typing import List, Dict, Any

OPENAI_CATEGORIES = [
    "hate", "hate/threatening", "harassment", "harassment/threatening",
    "self-harm", "self-harm/intent", "self-harm/instructions",
    "sexual", "sexual/minors", "violence", "violence/graphic",
]

S_CODE_TO_OPENAI = {
    "S1": ["violence"], "S2": ["violence"], "S3": ["sexual"], "S4": ["sexual"],
    "S5": ["harassment"], "S6": ["self-harm"], "S7": ["self-harm/intent"],
    "S8": ["self-harm/instructions"], "S9": ["violence"], "S10": ["hate"],
    "S11": ["self-harm/instructions"], "S12": ["hate/threatening"],
    "S13": ["harassment/threatening"], "S14": ["sexual/minors"],
}

def _init_zero_dict():
    return {k: False for k in OPENAI_CATEGORIES}, {k: 0.0 for k in OPENAI_CATEGORIES}

def make_openai_moderation_response(
    safe: bool,
    s_codes: List[str],
    explanation: str = "",
    raw: Dict[str, Any] = None
) -> Dict[str, Any]:
    categories_dict, scores = _init_zero_dict()

    # 根据 S-codes 填充 categories_dict 和 scores
    for s in (s_codes or []):
        mapped = S_CODE_TO_OPENAI.get(s.upper(), [])
        for cat in mapped:
            if cat in categories_dict:
                categories_dict[cat] = True
                scores[cat] = 1.0

    # 计算 flagged 状态
    flagged = not safe  # 如果 safe=False，则 flagged=True，表示有害内容

    # 构造 category_applied_input_types
    applied = {}
    for cat in OPENAI_CATEGORIES:
        if categories_dict[cat]:
            applied[cat] = ["text"]
        else:
            applied[cat] = []

    # 组装最终的返回结果
    result_payload = {
        "flagged": flagged,
        "categories": categories_dict,
        "category_scores": scores,
        "category_applied_input_types": applied,  # 确保正确返回
        "explanation": explanation or "",
        "raw": raw or {}
    }

    # 返回完整的响应
    return {
        "id": "modr-" + uuid.uuid4().hex,  # 使用动态生成的 ID
        "model": os.getenv("MODEL_NAME", "llama-guard-3-8b"),  # 使用环境变量配置模型名称
        "results": [result_payload]
    }
