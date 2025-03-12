# Ollama 和 AnythingLLM

本示例将基于Docker Compose部署Ollama和Open WebUI。

### 启动Ollama和AnythingLLM

首先为AnythingLLM容器引用的卷创建相应的目录，并将其属主和属组修改为AnythingLLM默认使用的“1000:1000”用户和组。

```bash
mkdir ./anythingllm
chown 1000:1000 ./anythingllm
```

随后，运行如下命令，即可启动这两个组件。

```bash
docker-compose pull  # 手动下载Image
docker-compose up -d   # 启动
```

上面的命令会创建并启动两个容器：ollama和AnythingLLM。

### 下载模型

运行如下命令，在ollama容器中下载模型，这里以DeepSeek蒸馏的Qwen小模型为例，它可以不依赖于GPU运行。

```bash
# 下载要部署的本地模型
docker-compose exec ollama ollama pull erwan2/DeepSeek-R1-Distill-Qwen-1.5B

# 下载用于词嵌入的模型
docker-compose exec ollama ollama pull nomic-embed-text:latest
```

### 访问AnythingLLM

访问入口是“http://HOST:3001”， 其中的HOST是Docker Compose所在的主机地址。

随后，在AnythingLLM上创建工作区，配置默认使用的“模型（model）”，即可进行使用。我们这里使用本地的Ollama上部署的模型，在工作区“**配置**”里找到“**聊天设置**”，在“**工作区 LLM 提供者**”选择“**Ollama**”，并在“**工作区聊天模型**”里选定要使用的本地模型即可。





