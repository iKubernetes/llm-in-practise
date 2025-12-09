# OpenAI Moderation Wrapper (litellm-compatible)



**OpenAI 审核封装器 (兼容 LiteLLM)的功能特性**

- /v1/moderations（以及 /moderations）接口返回 OpenAI 审核响应格式，该格式是 LiteLLM 代理 (litellm-proxy) 所预期的。

- 内部调用 vLLM 的 /v1/completions 接口，并使用 Llama-Guard 提示词模板（严格要求返回 JSON 格式）。

- 优先采用模型返回的 JSON 输出；如果返回的不是 JSON，则退而求其次，解析原生 "unsafe\nS1 S4"（不安全\nS1 S4）等格式。

- 将 Llama-Guard 的 S-代码（如 S1、S4）映射到 OpenAI 审核类别，并生成 flagged（已标记）和 category_scores（类别得分）。

- 支持简单的可选 X-API-KEY 检查（通过设置 WRAPPER_API_KEY 环境变量启用）。



**使用方法**

1. 将 Llama-Guard-3 模型权重文件放置在 ./models/guard 目录下，并确保 vLLM 能够加载它们。

2. 如果 vLLM 运行在其他位置，请修改 docker-compose.yml 文件并相应设置 MODEL_API_URL 环境变量。

3. 构建并运行：

   ```bash
   docker compose up --build
   ```

4. 测试：

   ```bash
   curl -X POST "http://localhost:8099/v1/moderations" -H "Content-Type: application/json" -d '{"input":"I want to harm someone"}'
   ```



**注意事项**

- 如果模型输出不稳定 (noisy) 或未返回 JSON，封装器将应用启发式规则（例如，检测 "自杀"、"bomb"、仇恨关键词）来促进 S-代码的匹配。

- 对于生产环境，请根据需要添加身份验证、TLS 反向代理、速率限制、日志存储和监控等措施。
