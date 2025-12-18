import time
import json
from locust import HttpUser, task, between

class LLMUser(HttpUser):
    wait_time = between(1, 3)  # 模拟用户思考时间

    @task
    def chat_completion(self):
        payload = {
            "model": "qwen3",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "请写一篇约200字的短文，介绍中国的长城。"}
            ],
            "max_tokens": 256,       # 控制输出长度，便于计算速率
            "temperature": 0.7,
            # "stream": False       # 默认非流式
        }

        start_time = time.time()
        
        with self.client.post("/v1/chat/completions", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # 输出 token 数（vLLM OpenAI 兼容接口返回）
                    output_tokens = data["usage"]["completion_tokens"]
                    
                    total_time = time.time() - start_time
                    # 减去 TTFB 近似（更准确的生成时间）
                    # 或者直接用 total_time 算整体吞吐
                    tokens_per_second = output_tokens / total_time if total_time > 0 else 0
                    
                    # 自定义事件上报到 Locust 统计（关键！）
                    self.environment.events.request.fire(
                        request_type="LLM",
                        name="tokens_per_second",
                        response_time=total_time * 1000,  # ms
                        response_length=output_tokens,
                        exception=None,
                        context={"tokens_per_second": tokens_per_second}
                    )
                    
                    # 可选：记录自定义指标
                    print(f"Generated {output_tokens} tokens in {total_time:.2f}s → {tokens_per_second:.2f} tokens/s")
                    
                except Exception as e:
                    response.failure(str(e))
            else:
                response.failure(f"HTTP {response.status_code}")
