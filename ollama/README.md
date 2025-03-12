# Ollama 和 Open WebUI

本示例将基于Docker Compose部署Ollama和Open WebUI。

### 启动Ollama和Open WebUI

运行如下命令，即可启动这两个组件。

```bash
docker-compose pull  # 手动下载Image
docker-compose up -d   # 启动
```

上面的命令会创建并启动两个容器：ollama和open-webui。

### 下载模型

运行如下命令，在ollama容器中下载模型，这里以DeepSeek蒸馏的Qwen小模型为例，它可以不依赖于GPU运行。

```bash
docker-compose exec ollama ollama pull erwan2/DeepSeek-R1-Distill-Qwen-1.5B
```

### 访问Open WebUI

访问入口是“http://HOST:3000”，其中的HOST是Docker Compose所在的主机地址。创建账号后即可例如使用。

需要注意的是，Open WebUI初始化的时间较长（需联网下载更新），首次访问需要耐心等待。


