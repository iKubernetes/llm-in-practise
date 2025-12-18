# Dockerfile
# 基于 vLLM 官方镜像
FROM vllm/vllm-openai:v0.11.2

# 安装 ray 库，并配置：
# 1. 使用阿里云镜像源 (https://mirrors.aliyun.com/pypi/simple/)
# 2. 不在本地缓存安装包 (--no-cache-dir)
RUN pip install ray[default,serve,vllm]==2.52.1 pyarrow gymnasium tree --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/
