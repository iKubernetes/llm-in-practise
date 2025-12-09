import os
import json
import re
import asyncio
from typing import Any, Dict
import httpx
from fastapi import HTTPException
import logging
from datetime import datetime

logger = logging.getLogger("model_client")

MODEL_API_URL = os.getenv("MODEL_API_URL", "http://llama-guard-vllm:8002")
COMPLETIONS_PATH = os.getenv("COMPLETIONS_PATH", "/v1/completions")
COMPLETIONS_URL = MODEL_API_URL.rstrip("/") + COMPLETIONS_PATH

# System prompt as strict instruction for Llama-Guard-3 classification
SYSTEM_PROMPT = (
    "You are Llama Guard 3, a safety classifier that must map text to a standard S-code taxonomy (S1..S14).\n"
    "You MUST respond in valid JSON ONLY (no additional text) with EXACT fields:\n"
    "  - \"safe\": boolean,\n"
    "  - \"categories\": an array of S-codes (e.g. [\"S1\"] or [\"S11\"]) - choose the most relevant codes in order of priority (top-3 max),\n"
    "  - \"explanation\": a short one-line justification why those categories apply.\n"
    "Rules:\n"
    "  1) Always pick S-codes from S1..S14. Up to 3 codes.\n"
    "  2) If the user asks for self-harm or suicide instructions, include S11.\n"
    "  3) If the user asks about making explosives or weapons, include S9 (Indiscriminate Weapons) or S2/S1 as appropriate.\n"
    "  4) If the user expresses hate toward a protected group, include S10.\n"
    "  5) NEVER output anything other than valid JSON.\n"
    "Example output:\n"
    "{\"safe\": false, \"categories\": [\"S11\"], \"explanation\": \"User asks how to commit suicide.\"}\n"
)

# low-level post with retries
async def _post_completions(prompt: str, model: str, max_tokens: int = 256, temperature: float = 0.0, timeout: int = 30, retries: int = 2) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_exc = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(retries + 1):
            try:
                resp = await client.post(COMPLETIONS_URL, json=payload)
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise HTTPException(status_code=502, detail=f"Model server error: {resp.status_code} {text.decode(errors='ignore')}")
                return resp.json()
            except httpx.RequestError as e:
                last_exc = e
                logger.warning("RequestError to model server (attempt %d/%d): %s", attempt + 1, retries + 1, e)
                await asyncio.sleep(0.5 * (attempt + 1))
            except HTTPException:
                raise
        raise HTTPException(status_code=502, detail=f"Error contacting model server after retries: {last_exc}")

def build_guard_prompt(user_text: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        + SYSTEM_PROMPT + "\n"
        + "<|start_header_id|>user<|end_header_id|>\n"
        + user_text.strip() + "\n"
        + "<|start_header_id|>assistant<|end_header_id|>\n"
    )

def parse_native_scodes(text: str):
    """
    Parse native outputs like:
      "unsafe\nS1 S4"
      "safe\nS0"
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return {"safe": True, "categories": [], "explanation": ""}
    first = lines[0].lower()
    safe = not ("unsafe" in first)
    # collect S-codes
    codes = re.findall(r"S\d+", " ".join(lines[1:]) or " ".join(lines))
    codes = [c.upper() for c in codes]
    if not codes and not safe:
        codes = ["UNSPECIFIED"]
    return {"safe": safe, "categories": codes, "explanation": ""}

async def vllm_moderation(text: str, model: str) -> Dict[str, Any]:
    prompt = build_guard_prompt(text)
    resp = await _post_completions(prompt, model=model, max_tokens=256, temperature=0.0, timeout=30, retries=2)

    # Extract text robustly
    try:
        choice = resp.get("choices", [])[0]
        text_out = choice.get("text") or choice.get("message", {}).get("content") or ""
    except Exception:
        text_out = json.dumps(resp)

    # 1) Prefer JSON output from model
    text_out_stripped = (text_out or "").strip()
    try:
        parsed = json.loads(text_out_stripped)
        # validate categories
        cats = parsed.get("categories", []) or []
        cats = [c.upper() for c in cats if isinstance(c, str)]
        parsed_safe = bool(parsed.get("safe", True))
        parsed_expl = parsed.get("explanation", "") or ""
        return {"safe": parsed_safe, "categories": cats, "explanation": parsed_expl, "raw": {"text": text_out_stripped, "resp": resp}}
    except Exception:
        # fallback: parse native S-codes
        parsed_native = parse_native_scodes(text_out_stripped)
        # heuristic augmentations: quick keyword rules
        low = text.lower()
        if ("suicid" in low) or ("自杀" in low):
            if "S11" not in parsed_native["categories"]:
                parsed_native["categories"].insert(0, "S11")
                parsed_native["safe"] = False
                parsed_native["explanation"] = "Heuristic: contains suicide/self-harm keywords."
        elif any(k in low for k in ["bomb", "炸弹", "explosive", "爆炸"]):
            if "S9" not in parsed_native["categories"]:
                parsed_native["categories"].insert(0, "S9")
                parsed_native["safe"] = False
                parsed_native["explanation"] = "Heuristic: contains weapon/explosive keywords."
        elif any(k in low for k in ["hate", "我讨厌", "黑人", "仇恨", "racist", "杀死他们因为"]):
            if "S10" not in parsed_native["categories"]:
                parsed_native["categories"].insert(0, "S10")
                parsed_native["safe"] = False
                parsed_native["explanation"] = "Heuristic: contains hate-speech keywords."
        return {"safe": parsed_native["safe"], "categories": parsed_native["categories"], "explanation": parsed_native["explanation"], "raw": {"text": text_out_stripped, "resp": resp}}
