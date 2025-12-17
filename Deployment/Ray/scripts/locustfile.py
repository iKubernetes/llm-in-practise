from locust import HttpUser, task, between

# 模拟 Qwen3 请求体
QWEN3_CHAT_PAYLOAD = {
    "model": "qwen3",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        # 模拟一个相对较长的输入，以匹配 --input-len 1024 的效果
        {"role": "user", "content": "请写一个关于大型语言模型自动扩缩容的详细技术报告，长度约为1024个 token 的提示词。"}
    ],
    "max_tokens": 128,  # 匹配 --output-len 128
    "temperature": 0.0,
    "stream": False
}

class LLMUser(HttpUser):
    # 匹配您的配置: 目标 QPS 10，所以我们将等待时间设置在 0.1 秒左右
    wait_time = between(0.05, 0.15) 
    host = "http://127.0.0.1:8000"

    @task
    def chat_completion(self):
        # 发送请求到 Ray Serve 代理的 chat completion 接口
        self.client.post("/v1/chat/completions", json=QWEN3_CHAT_PAYLOAD)
