#!/bin/bash

VLLM_URL="http://192.168.70.54:8000/v1/completions"

# 构造一个较长的共享前缀（约 800~1000 tokens）
SHARED_PREFIX="Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts models developed by Alibaba Cloud. Qwen3 introduces several groundbreaking features including native reasoning capabilities, multi-language support across 29 languages, advanced agentic abilities with tool use and code execution, and a novel dual-mode thinking mechanism that allows seamless switching between fast intuitive responses and deep deliberate reasoning. The model family includes both dense models ranging from 0.6B to 32B parameters and mixture-of-experts variants with up to 235B total parameters and 32B active parameters. Qwen3 demonstrates state-of-the-art performance on benchmarks including MMLU, GPQA, and GSM8K, while maintaining high efficiency through architectural innovations such as grouped query attention, sliding window attention, and expert routing optimization. The training process involved pre-training on over 30 trillion tokens followed by supervised fine-tuning and reinforcement learning from human feedback. Qwen3 supports context lengths up to 128K tokens and offers advanced capabilities in mathematics, coding, logical reasoning, and creative writing. "

# 函数：发送请求并提取 TTFT 和首段文本
test_request() {
    local label="$1"
    local prompt_suffix="$2"
    
    echo "========== $label =========="
    
    # 将时间数据写入临时文件，响应体输出到 stdout
    local timing_file=$(mktemp)
    
    response=$(curl -s -w "\n%{time_starttransfer}\n%{http_code}" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"qwen3-8b\",
            \"prompt\": \"${SHARED_PREFIX}${prompt_suffix}\",
            \"max_tokens\": 100,
            \"temperature\": 0.7
        }" \
        "$VLLM_URL")
    
    # 分离：最后一行是 HTTP code，倒数第二行是 TTFT，前面是 JSON
    http_code=$(echo "$response" | tail -n1)
    ttft=$(echo "$response" | tail -n2 | head -n1)
    json_body=$(echo "$response" | sed '$d' | sed '$d')
    
    echo "HTTP_CODE: $http_code"
    echo "TTFT: ${ttft}s"
    
    # 解析首段文本
    text=$(echo "$json_body" | jq -r '.choices[0].text' 2>/dev/null | head -c 150)
    echo "Response: $text..."
    echo ""
}

# 测试 1：冷启动（Cache Miss）
test_request "测试 1：冷启动（Cache Miss）" "Please summarize the key features of Qwen3 in three bullet points."

# 测试 2：相同前缀（预期 Cache Hit）
test_request "测试 2：相同前缀（预期 Cache Hit）" "What are the main architectural improvements in Qwen3 compared to Qwen2.5?"

# 测试 3：完全不同前缀（预期 Cache Miss）
echo "========== 测试 3：完全不同前缀（预期 Cache Miss） =========="
response=$(curl -s -w "\n%{time_starttransfer}\n%{http_code}" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-8b",
        "prompt": "Tell me a joke about Kubernetes and containers.",
        "max_tokens": 100,
        "temperature": 0.7
    }' \
    "$VLLM_URL")

http_code=$(echo "$response" | tail -n1)
ttft=$(echo "$response" | tail -n2 | head -n1)
json_body=$(echo "$response" | sed '$d' | sed '$d')

echo "HTTP_CODE: $http_code"
echo "TTFT: ${ttft}s"
text=$(echo "$json_body" | jq -r '.choices[0].text' 2>/dev/null | head -c 150)
echo "Response: $text..."
