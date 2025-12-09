FROM python:3.11-slim

WORKDIR /app

# 安装需要的依赖
COPY requirements.txt /app/
RUN pip install -r requirements.txt --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/
# 若无需使用阿里云的pip代理，则使用如下这行来代替上面的那行
#RUN pip install -r requirements.txt --no-cache-dir

COPY . /app

# 默认监听的端口（容器内部）
ENV WRAPPER_PORT=8099
# 后端的LlaMA-Guard-3服务的访问端点和模型名称
ENV MODEL_API_URL="http://guard-engine:8002"
ENV MODEL_NAME="llama-guard-3"

EXPOSE ${WRAPPER_PORT}

CMD ["python", "app.py"]

