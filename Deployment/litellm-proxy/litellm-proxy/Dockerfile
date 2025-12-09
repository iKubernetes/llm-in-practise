# Dockerfile
# 基于 LiteLLM 官方镜像
FROM ghcr.io/berriai/litellm:main-latest

# 安装 redisvl 库，并配置：
# 1. 使用阿里云镜像源 (https://mirrors.aliyun.com/pypi/simple/)
# 2. 不在本地缓存安装包 (--no-cache-dir)
RUN pip install redisvl --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/
