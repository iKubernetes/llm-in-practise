# OpenAI Moderation Wrapper (litellm-compatible)

**Llama-Guard-3 -> OpenAI Moderation Wrapper (litellm-compatible)**



**Features:**

- /v1/moderations (and /moderations) returns OpenAI moderation response shape expected by litellm-proxy.
- Internally calls vLLM /v1/completions with Llama-Guard prompt template (strict JSON requested).
- Prefers model JSON output; falls back to parsing native "unsafe\\nS1 S4" format.
- Maps S-codes -> OpenAI categories and produces flagged + category_scores.
- Simple optional X-API-KEY check (set WRAPPER_API_KEY env).



**Usage:**

1) Put your Llama-Guard-3 weights under ./models/guard and ensure vLLM is able to load them.
2) Edit docker-compose.yml if your vLLM runs elsewhere and set MODEL_API_URL accordingly.
3) Build & run:
   docker compose up --build

4) Test:
   curl -X POST "http://localhost:8099/v1/moderations" -H "Content-Type: application/json" -d '{"input":"I want to harm someone"}'



**Notes:**

- If model is noisy or returns non-JSON, wrapper applies heuristics (e.g., detect "自杀", "bomb", hate keywords) to promote matching S-codes.
- For production, add authentication, TLS reverse proxy, rate limits, logging storage and monitoring as needed.
