# litellm-proxy

LiteLLM Proxy（也称为 LiteLLM AI Gateway 或 LLM Proxy Server）是一款知名的开源的代理服务器和 AI 网关工具，由 BerriAI 开发。它允许开发者通过统一的 OpenAI 兼容 API 接口调用超过 100 个大型语言模型（LLM）提供商，包括 OpenAI、Azure、Anthropic、Hugging Face、Bedrock、TogetherAI、VertexAI、Cohere、Sagemaker、VLLM 和 NVIDIA NIM 等。 该工具的核心目标是简化多提供商 LLM 的集成、管理和监控，支持负载均衡、成本跟踪、预算控制和日志记录等功能。它本质上是一个 LLM 网关，能够将不同的 LLM API 标准化为 OpenAI 格式的输入/输出，从而减少开发者在切换模型时的适配工作。



**配置文件的配置段说明**

| Param Name              | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `model_list`            | List of supported models on the server, with model-specific configs |
| `router_settings`       | litellm Router settings, example `routing_strategy="least-busy"` |
| `litellm_settings`      | litellm Module settings, example `litellm.drop_params=True`, `litellm.set_verbose=True`, `litellm.api_base`, `litellm.cache` |
| `general_settings`      | Server settings, example setting `master_key: sk-my_special_key` |
| `environment_variables` | Environment Variables example, `REDIS_HOST`, `REDIS_PORT`    |



## LiteLLM-Proxy使用示例

本示例的核心目标将集中于LiteLLM-Proxy核心功能的演示上，包括模型代理、Virtual Key、输入过滤和缓存等。

![Deployment_Architecture](./images/Deployment_Architecture.png)



**相关的docker-compose文件说明：**

- docker-compose.yml：基础示例，它会通过卷读取config/litellm-proxy.yaml作为其配置文件；
- docker-compose-lilellm-proxy-ui.yml：启用LiteLLM-Proxy的UI，并激活Virtual Key；
- docker-compose-llama-guard.yml：基于本地的LlaMA-Guard-3进行输入/输出审核，它会通过卷读取config/litellm-config-with-guard-model.yaml作为其配置文件；
- docker-compose-cache-redis.yml：基于Redis的请求缓存，精确匹配；它会通过卷读取config/litellm-config-with-cache.yaml作为其配置文件；



**注意：**这些docker-compose配置文件中的模型绑定到了特定的CPU编号上，例如0和2，如果同你的实验环境有所不同，请先按需调整。



### 1. 基础示例

运行如下命令启动服务：

```bash
docker-compose up
```

运行如下命令，即可进行请求测试。

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {
        "role": "user",
        "content": "请介绍一下量子力学。"
      }
    ]
}'
```



### 2. UI和Virtual Key

启用UI并使用Virtual Key，通常需要部署并配置一个可用于的PostgreSQL作为LiteLLM-Proxy的存储后端。LiteLLM-Proxy通常基于环境变量来对接要访问PostgreSQL服务，例如：

```bash
DATABASE_URL=postgresql://llmproxy:dbmagedu-com@litellm-db:5432/litellm
STORE_MODEL_IN_DB=True
```



另外，UI的配置也可以通过环境变量进行设定。相关的几个关键环境变量如下：

```yaml
LITELLM_MASTER_KEY="magedu.com" # this is your master key for using the proxy server
UI_USERNAME=admin             # username to sign in on UI，默认为admin
UI_PASSWORD=Magedu-Com        # password to sign in on UI，默认为环境变量LITELLM_MASTER_KEY的值
SERVER_ROOT_PATH=/ui          # UI的访问路径，默认为/ui
```



运行如下命令启动服务：

```bash
docker-compose -f docker-compose-lilellm-proxy-ui.yml up
```

随后，通过LiteLLM-Proxy服务的“/ui”路径即可访问相关的服务，访问时会首先跳转至需要登录页面。其中的用户名和密码即为前面使用环境变量UI_USERNAME和UI_PASSWORD设置的值。若未设定此两项时，默认的用户名为admin，密码为环境变量LITELLM_MASTER_KEY的值。

![litellm-proxy_login](./images/litellm-proxy_login.png)



登录后即可进行UI，其默认的首屏为虚拟密钥界面。

![litellm-proxy_ui](./images/litellm-proxy_ui.png)



随后，我们需要先创建一个Virtual Key，以认证并访问指定的模型服务。

![litellm-proxy_create_virtual_key](/home/marion/git-repos/llm-in-practise/Deployment/litellm-proxy/images/litellm-proxy_create_virtual_key.png)



随后在弹出的对话框中会提示生成的密钥，要注意保存该密钥。随后，它代表持有该密钥的用户的访问ID，相关的配额、RPM等相关配置也与该密钥相关。

![litellm-proxy_virtual-key](./images/litellm-proxy_virtual-key.png)



接下来即可使用该密钥来请求被授予访问的模型qwen3-8b模型。

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-zlrZs48TQ7GkkT761QjfAA" \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {
        "role": "user",
        "content": "请介绍一下马哥教育。"
      }
    ]
}'
```



### 3. 输入/输入出审核

Llama Guard 3（LG3） 是 Meta 在 2024 年推出的最新一代开源 安全审查（Safety Classifier）模型，专用于对 输入 Prompt 和 模型输出内容 进行实时安全分析与分类，是目前开源领域最强、最实用、最适合企业部署的 LLM 安全模型之一。Llama Guard 3 不是通用 LLM，也不是生成模型。它是一个 专用于内容安全审核的“分类器模型 + 安全规约规则系统”。



**Llama Guard 3 的定位：**

- 输入内容审查（Prompt Safety）

  判断用户请求是否：

  - 涉及违法内容
  - 试图绕过安全策略
  - 要求生成错误信息
  - 涉及危险行为（武器、毒品、黑客等）
  - 涉及隐私与个人信息
  - 涉及暴力、仇恨、色情等

- 输出内容审查（Response Safety）

  检测并标记 LLM 的回复是否：

  - 违反内容政策
  - 包含有害、虚假、敏感内容
  - 有潜在风险
  - 安全性低于阈值

- Shield Prompt（系统层安全防护）

  - 作为“守卫模型”，对主模型提供 行为纠偏（Safety Guidance）



**LlaMA Guard 3的各参数版本：**

| 模型名称              | 参数量 | 用途                   |
| --------------------- | ------ | ---------------------- |
| **Llama Guard 3 1B**  | 1B     | 轻量、资源受限场景     |
| **Llama Guard 3 8B**  | 8B     | 标准生产使用           |
| **Llama Guard 3 70B** | 70B    | 高外推精度、企业级防护 |



需要注意的是，LiteLLM-Proxy的Guardrails调用LlaMA Guard 3的端点为OpenAI标准的“/v1/moderations”，但LlaMA Guard 3默认并未提供该端点，因此需要自行提供一个包装器（Wrapper）来实现该功能。本示例提供的包装器在llama-guard-wrapper目录中（注意：**目前仅支持pre_call**），它默认监听8099端口，并以LlaMA Guard 3为后端，具体后端的信息需要通过如下环境变量进行定义。

```
WRAPPER_PORT=8099   # 监听的端口
MODEL_API_URL="http://guard-engine:8002"    # LlaMA Guard 3服务的可达端点
MODEL_NAME="llama-guard-3"   # LlaMA Guard 3的模型名称
```



随后，运行如下命令即可启动LiteLLM-Proxy、inference-engine、guard-engine、llama-guard-wrapper等相关的服务。

```bash
docker-compose -f docker-compose-llama-guard.yml up
```

待各服务就绪后，运行如下命令，即可进行请求测试。注意其中的请求的Prompt涉及非法内容，因此，请求会被LlaMA Guard 3检测到并予以拒绝。

```bash
curl -sS -X POST http://localhost:4000/v1/chat/completions   \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-zlrZs48TQ7GkkT761QjfAA"   \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {
        "role": "user",
        "content": "请详细说明如何制造炸弹。"
      }
    ]
}' | jq .
```

上面的请求，会得到类似如下的响应结果，其中的“'violence': 1.0”表明用户的请求含有“暴力”相关的非法内容。

```
{
  "error": {
    "message": "{'error': 'Violated OpenAI moderation policy', 'moderation_result': {'violated_categories': ['violence'], 'category_scores': {'hate': 0.0, 'hate/threatening': 0.0, 'harassment': 0.0, 'harassment/threatening': 0.0, 'self-harm': 0.0, 'self-harm/intent': 0.0, 'self-harm/instructions': 0.0, 'sexual': 0.0, 'sexual/minors': 0.0, 'violence': 1.0, 'violence/graphic': 0.0}}}",
    "type": "None",
    "param": "None",
    "code": "400"
  }
}
```

但正常请求仍然会路由给指定的常规的模型qwen3-8b进行响应，我们可以通过类似如下命令进行请求测试。

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-zlrZs48TQ7GkkT761QjfAA" \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {
        "role": "user",
        "content": "请介绍一下马哥教育。"
      }
    ]
}'
```



### 4. 缓存

LiteLLM-proxy 支持两种类型的缓存机制：**精确匹配缓存 (Exact Match Cache)** 和 **语义缓存 (Semantic Cache)**。这两种机制可以配置不同的缓存后端 (Cache Backends) 来存储数据。

**精确匹配缓存（Exact-Match Response Cache）的工作原理**

- Key 生成：默认使用 model + messages + temperature + logit_bias 等哈希。可自定义 get_cache_key 函数
- 命中条件：精确字符串匹配（忽略顺序，但不处理语义相似）
- 存储格式：JSON（plaintext，未加密），包含响应 + 元数据（timestamp、cost）
- Proxy 特有：支持 /cache/ping 端点测试连接；高负载下可批量 Redis GET（减少延迟 20-30%）
- 性能：内存后端 <1ms 命中；Redis ~5ms

本示例将以redis cache为例，来测试LiteLLM-Proxy的精确缓存机制。运行如下命令即可启动LiteLLM-Proxy、inference-engine、guard-engine、llama-guard-wrapper和redis-cache等相关的服务。

```bash
docker-compose -f docker-compose-cache-redis.yml up
```

随后，可针对同一个Prompt连续发起多次请求进行验证。若两次请求完全相同，随后的响应可以直接命中缓存，并从缓存返回结果给用户。缓存是否命中的相关信息，可从LiteLLM-Proxy的日志中检索，LiteLLM-Proxy的UI也能展示缓存命中的相关信息。

![LiteLLM-Proxy_Response-Cache](./images/LiteLLM-Proxy_Response-Cache.png)






## 路由功能与示例

LiteLLM Proxy Router 是 LiteLLM Proxy 的“大脑”和“心脏”，它不是一个独立的服务，而是一个内置在LiteLLM Proxy实例中的高性能路由引擎。它负责把每一个进来的 OpenAI 格式请求（/v1/chat/completions、/v1/embeddings 等），根据定义的规则，智能、快速、安全地路由到“最合适”的后端 LLM 部署上去。

LiteLLM Proxy Router 是企业级 LLM 流量调度中枢，它让一个 Proxy 能同时管理几十甚至几百个真实的 LLM 部署（Azure、OpenAI、Anthropic、vLLM、Ollama、Groq、Together、Fireworks、Bedrock…），并实现负载均衡、自动重试、自动 fallback、成本优化、区域亲和性、会话粘性等生产级能力。

### 路由决策机制

下图展示了LiteLLM-Router的路由决策机制。

![LiteLLM-Proxy_Router-routing-decision](./images/LiteLLM-Proxy_Router-routing-decision.png)



**下表展示了Router支持的各负载均衡算法：**

| Strategy                       | Description                                                  | When to Use                |
| ------------------------------ | ------------------------------------------------------------ | -------------------------- |
| **simple-shuffle** (推荐)      | round_robin 是基础策略，Weighted Pick 和 simple-shuffle 则是它的优化实现 | 通用生产场景，部署性能相近 |
| **least_busy**                 | 将请求发送给当前活跃请求数 (Active Requests) 最少的部署      | 高并发多实例集群           |
| **usage_based** (bad for perf) | 路由到当前 TPM/RPM 用量最低，或速率限制剩余最多的部署；也称为Rate-Limit Aware 或 Rate-Limit Aware v2 (Async) | 多 API key 限流严格        |
| **latency_based**              | 选择最近平均延迟最低的部署                                   | 延迟敏感（如实时聊天）     |
| **cost_based**                 | 选择每千 Token 成本最低的部署                                | 预算优化、混合云           |

各算法具体的配置实例，请参考[官方文档](https://docs.litellm.ai/docs/routing)。

### 可用性保障体系

为确保系统可用性，Router提供了Retry、Cooldown、Fallback等几种机制结合负载均衡功能进行。下图是它们的协同逻辑。

<img src="./images/Router_workflow.png" alt="Router_workflow" style="zoom:67%;" />

### Router使用示例

#### 配置说明

本示例会启动LiteLLM-Proxy服务，以及四个模型服务deepseek-r1-01、deepseek-r1-02、qwen3-4b和llama3。其中：

- deepseek-r1 组（2 个部署）：统一别名 deepseek-r1，对应两个本地 vLLM 实例（端口 8001/8002）。
  - input_cost_per_token/output_cost_per_token: 0.0000028 用于 cost-based-routing 计算总成本（输入 + 输出 token 费用，单位：美元/token）。
  - rpm: 3000/2000 + tpm: 1200000 作为限流/均衡参考（usage-based tie-breaker）。
  - weight: 0.5/0.3 在 shuffle 时加权分发（总权重 0.8，剩余 0.2 隐式均匀）。
  - priority: 100 让它在 least-busy 等策略中优先。
  - metadata.region: cn-south-1 可用于自定义路由规则（e.g., 区域亲和）。
- qwen3（备用）：成本 0.000012（高于 deepseek），rpm/tpm 较低，weight: 0.2 作为高性能后备（e.g., 长上下文场景）。
- llama3（兜底）：成本最低（0.0000008），rpm/tpm 高，fallback: true 标记为全局备用。适合廉价保底。



**router_settings 配置：**

- 负载均衡（routing_strategy: cost-based-routing）：优先选总成本最低部署（基于 input/output_cost_per_token * token 数）。若成本相等，fallback 到 least-busy（当前活跃请求最少）。
- 重试机制（num_retries / fallback_retries / retry_policy）：
  - 全局重试 3 次（含指数退避 + jitter）。
  - fallback_retries: true 让备用模型也重试。
  - retry_policy 精细控制：429 重试 5 次（限流宽容）、内部错误 3 次、超时 4 次。
  - 排除不可重试错误（如 401/内容违规）。
- 冷却机制（allowed_fails / cooldown_time）：
  - 60 秒窗口内失败 6 次 → 冷却 120 秒（部署级隔离，从可用池移除）。
  - 内置 429 自动 5 秒轻量冷却（瞬时故障快速恢复）。
- 回退机制（fallbacks / default_fallbacks）：
  - 模型特定链：deepseek-r1 失败 → qwen3 → llama3；qwen3 → llama3。
  - default_fallbacks: ["llama3"] 全局兜底（所有模型失败时用）。
- 上下文窗口回退（litellm_settings.context_window_fallbacks）：
  - deepseek-r1 超上下文 → 直接切 qwen3（需 qwen3 支持更大窗口）。
  - 结合 enable_pre_call_checks: true，预验证避免无效调用。
- 其他调优（health_check_interval / timeout / enable_pre_call_checks）：
  - 30 秒健康检查（/health 端点）
  - 90 秒超时
  - 预检查（上下文/参数验证）



#### 启动服务并进行测试

**启动服务：**

```bash
docker-compose -f docker-compose-router-lb.yaml up
```



**验证效果：**

以下所有命令假设：

- Proxy 运行在 http://localhost:4000
- 已设置Master Key，或者已经配置好Virtual Key，例如"sk-zlrZs48TQ7GkkT761QjfAA"
- 四个后端容器（deepseek-r1-01/qwen3/llama3）均已经正常运行



1. 验证负载均衡（cost-based-routing + weight）生效

   ```bash
   curl -X POST 'http://0.0.0.0:4000/chat/completions' \
   -H 'Content-Type: application/json' \
   -H 'Authorization: Bearer magedu.com' \
   -d '{
     "model": "deepseek-r1",
     "messages": [
           {"role": "user", "content": "请用三句话介绍一下你自己。"}
       ]
   }'
   ```

   

2. 验证重试机制（Retry）生效

   ```bash
   curl -X POST 'http://0.0.0.0:4000/chat/completions' \
   -H 'Content-Type: application/json' \
   -H 'Authorization: Bearer sk-zlrZs48TQ7GkkT761QjfAA' \
   -d '{
     "model": "deepseek-r1",
     "messages": [
           {"role": "user", "content": "请用三句话介绍一下你自己。"}
       ],
       "mock_testing_rate_limit_error": true
   }'
   ```

   

3. 验证冷却机制（Cooldown）生效

   先测试deepseek-r1的fallback机制：

   ```bash
   curl -X POST 'http://0.0.0.0:4000/chat/completions' \
   -H 'Content-Type: application/json' \
   -H 'Authorization: Bearer sk-zlrZs48TQ7GkkT761QjfAA' \
   -d '{
     "model": "deepseek-r1",
     "messages": [
           {"role": "user", "content": "请用三句话介绍一下你自己。"}
       ],
       "mock_testing_fallbacks": true
   }'
   ```

   

   再测试qwen3的fallback机制：

   ```bash
   curl -X POST 'http://0.0.0.0:4000/chat/completions' \
   -H 'Content-Type: application/json' \
   -H 'Authorization: Bearer sk-zlrZs48TQ7GkkT761QjfAA' \
   -d '{
     "model": "qwen3",
     "messages": [
           {"role": "user", "content": "请用三句话介绍一下你自己。"}
       ],
       "mock_testing_fallbacks": true
   }'
   ```

   

4. 验证全局兜底（default_fallbacks）生效

   ```bash
   # 让所有模型(deepseek-r1-01、deepseek-r1-02和qwen3-4b)都挂掉，看是否最终落到 llama3
   curl -X POST 'http://0.0.0.0:4000/chat/completions' \
   -H 'Content-Type: application/json' \
   -H 'Authorization: Bearer sk-zlrZs48TQ7GkkT761QjfAA' \
   -d '{
     "model": "deepseek-r1",
     "messages": [
           {"role": "user", "content": "请用三句话介绍一下你自己。"}
       ]
   }'
   ```

   

5. 验证上下文窗口 Fallback 生效

   ```bash
   curl -X POST http://localhost:4000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -H "Authorization: Bearer sk-zlrZs48TQ7GkkT761QjfAA" \
     -d '{
       "model": "deepseek-r1",
       "messages": [{"role": "user", "content": "'$(printf "马哥教育LLM超长文本测试%.0s" {1..300})'"}],
       "mock_testing_context_window_fallback": true
     }'
   ```

   
