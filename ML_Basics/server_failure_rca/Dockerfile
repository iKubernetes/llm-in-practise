# 使用 Python 3.8 轻量镜像作为基础
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 创建必要的目录
RUN mkdir -p data/raw data/processed models scripts src logs config

# 复制项目文件
COPY models/* models/
COPY sciprts/* scripts/
COPY config/config.yaml config/config.yaml
#COPY src/* src/

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config/config.yaml

# 暴露 FastAPI 服务端口
EXPOSE 8000

# 运行 FastAPI 服务
CMD ["uvicorn", "scripts.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
