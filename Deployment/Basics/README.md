# 容器化vLLM



### 安装Docker

```bash
# step 1: 安装必要的一些系统工具
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg

# step 2: 信任 Docker 的 GPG 公钥
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Step 3: 写入软件源信息
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
 
# Step 4: 安装Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 安装指定版本的Docker-CE:
# Step 1: 查找Docker-CE的版本:
# apt-cache madison docker-ce
#   docker-ce | 17.03.1~ce-0~ubuntu-xenial | https://mirrors.aliyun.com/docker-ce/linux/ubuntu xenial/stable amd64 Packages
#   docker-ce | 17.03.0~ce-0~ubuntu-xenial | https://mirrors.aliyun.com/docker-ce/linux/ubuntu xenial/stable amd64 Packages
# Step 2: 安装指定版本的Docker-CE: (VERSION例如上面的17.03.1~ce-0~ubuntu-xenial)
# sudo apt-get -y install docker-ce=[VERSION]

```



### 命令行启动



```bash
docker run --runtime nvidia --gpus all \
   -v ~/.cache/huggingface:/root/.cache/huggingface -v /Models/Pretrained_Models:/data/models  \
   --ipc=host  -p 8000:8000  vllm/vllm-openai:v0.11.2  --model "/data/models/Qwen3-8B"  \
   --trust-remote-code     --max-model-len 8192  --served-model-name "Qwen3-8B"
```



启动成功后，vLLM API Server 会在 http://localhost:8000 上运行，随后我们可以使用 cURL 或任何 HTTP 客户端向 /v1/completions 或 /v1/chat/completions 端点发送请求。

```
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B",
    "prompt": "请介绍一下马哥教育。",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```



流式输出：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B",
    "messages": [
      {
        "role": "user",
        "content": "请用五句话介绍一下马哥教育。"
      }
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": true 
  }'
```



### 推荐的 docker-compose.yml 模板

锁定容器化vLLM最佳实践的示例模板

```yaml
version: '3.9'

services:
  vllm-inference:
    image: vllm/vllm-openai:v0.11.2  # 1. 锁定具体版本
    container_name: vllm_service
    runtime: nvidia
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} # 通过 .env 文件传入 Token
      - CUDA_VISIBLE_DEVICES=0 # 指定 GPU
    volumes:
      - ./models:/data/models # 2. 挂载本地模型路径
      - ~/.cache/huggingface:/root/.cache/huggingface # 共享 HF 缓存
    ports:
      - "8000:8000"
    ipc: host # 3. 优化多卡通信性能（单卡也可用）
    command: >
      --model /data/models/Qwen3-8B
      --dtype bfloat16
      --gpu-memory-utilization 0.85
      --max-model-len 8192
      --max-num-seqs 256
      --enforce-eager # 某些特定架构可能需要，视情况而定
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck: # 4. 健康检查
      test: ["CMD-SHELL", "curl -f http://localhost:8000/v1/models || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: unless-stopped
```



### vLLM和Open WebUI示例

```yaml
version: '3.9'

services:
  # ----------------------------------------
  # 1. 推理后端: vLLM
  # ----------------------------------------
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm_service
    runtime: nvidia
    restart: unless-stopped
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} # 如果需要下载受限模型，请在 .env 文件中配置
      - CUDA_VISIBLE_DEVICES=0 # 指定 GPU
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface # 挂载宿主机缓存，避免重复下载
      - ./models:/data/Pretrained_Models
    ports:
      - "8000:8000"
    ipc: host # 优化内存通信
    # --- 启动命令详解 ---
    # --model: 模型名称 (HuggingFace 上的 repo id)，建议事先下载模型
    # --api-key: 设置 API 密钥，WebUI 连接时需要
    # --gpu-memory-utilization: 预留一部分显存给系统，防止 OOM
    command: >
      --model Qwen3/Qwen3-8B
      --gpu-memory-utilization 0.85
      --max-model-len 8192
      --dtype auto
      --api-key vllm-secret-key
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1 # 使用的 GPU 数量
              capabilities: [gpu]

  # ----------------------------------------
  # 2. 前端界面: Open WebUI
  # ----------------------------------------
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open_webui
    restart: unless-stopped
    ports:
      - "8080:8080" # 浏览器访问端口 (宿主机 8080 -> 容器 8080)
    volumes:
      - ./data/open-webui:/app/backend/data # 持久化聊天记录和用户数据
    environment:
      # --- 连接配置 ---
      # 告诉 WebUI 去哪里找 vLLM (注意: 使用容器服务名 'vllm')
      - OPENAI_API_BASE_URL=http://vllm:8000/v1
      - OPENAI_API_KEY=vllm-secret-key
      # --- 可选配置 ---
      - WEBUI_NAME=My Local AI
      # 如果是单人使用，可以关闭注册功能，只允许第一个创建的账号登录
      # - ENABLE_SIGNUP=False 
    depends_on:
      - vllm # 等待 vLLM 服务启动
```

