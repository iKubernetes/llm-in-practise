import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
from schemas import ModerationRequest
from model_client import vllm_moderation
from openai_moderation_map import make_openai_moderation_response

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("llama-guard-wrapper")

app = FastAPI(title="Llama-Guard-3 OpenAI-Moderation Wrapper")

WRAPPER_PORT = int(os.getenv("WRAPPER_PORT", "8099"))
MODEL_NAME = os.getenv("MODEL_NAME", "llama-guard-3")
# Basic simple token auth (optional). If set, incoming requests must include header X-API-KEY match.
API_KEY = os.getenv("WRAPPER_API_KEY", "")

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if API_KEY:
        key = request.headers.get("X-API-KEY", "")
        if key != API_KEY:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)

@app.post("/v1/moderations")
@app.post("/moderations")
async def moderations(req: ModerationRequest):
    """
    Accepts {"input": "..."} and returns OpenAI-compatible moderation response.
    Internally calls vLLM /v1/completions (MODEL_API_URL) with Llama-Guard prompt template.
    """
    text = req.input
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="input is required")

    logger.info("Moderation request received (len=%d)", len(text))
    # Call model client
    try:
        model_result = await vllm_moderation(text, model=MODEL_NAME)
    except HTTPException as e:
        logger.error("Model client error: %s", e.detail)
        raise
    except Exception as e:
        logger.exception("Unexpected error calling model client")
        raise HTTPException(status_code=502, detail=str(e))

    # model_result expected shape:
    # {"safe": bool, "categories": ["S1","S11"], "explanation": "...", "raw": {...}}
    safe = bool(model_result.get("safe", True))
    s_codes = model_result.get("categories", []) or []
    explanation = model_result.get("explanation", "") or ""
    raw = model_result.get("raw", {})

    # Build OpenAI moderation response
    openai_resp = make_openai_moderation_response(safe=safe, s_codes=s_codes, explanation=explanation, raw=raw)
    logger.debug("Returning moderation response: flagged=%s categories=%s", openai_resp["results"][0]["flagged"], list(openai_resp["results"][0]["categories"].keys()))
    return JSONResponse(content=openai_resp)

@app.post("/v1/generate")
async def generate(request: Request):
    # basic pass-through generation for convenience (calls completions endpoint)
    body = await request.json()
    return JSONResponse(status_code=501, content={"error": "generate not implemented in this wrapper. Use /v1/moderations for safety classification."})

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting wrapper on port %d (MODEL_API_URL=%s)", WRAPPER_PORT, os.getenv("MODEL_API_URL"))
    uvicorn.run("app:app", host="0.0.0.0", port=WRAPPER_PORT, log_level="info")
