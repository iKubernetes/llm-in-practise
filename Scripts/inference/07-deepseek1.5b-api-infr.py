import os
import time
import uuid
import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Any

from modelscope import AutoTokenizer, AutoModelForCausalLM
from modelscope.hub.snapshot_download import snapshot_download

# --- 1. 模型准备 ---
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 全局加载模型和分词器
try:
    model_dir = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("模型加载完成并进入评估模式。")
except Exception as e:
    print(f"模型加载失败: {e}")
    # 在生产环境中，可以考虑在这里退出或设置标志位，使应用无法启动
    tokenizer = None
    model = None

# --- 2. Prompt构造函数 (适应OpenAI messages格式) ---
# 修正类型提示：messages 实际上是 ChatMessage 对象的列表
def build_prompt_from_openai_messages(messages: List['ChatMessage']) -> str:
    """
    将OpenAI API的messages格式转换为DeepSeek模型所需的prompt格式。
    messages 示例: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    """
    prompt = ""
    for message in messages:
        # 直接通过点运算符访问属性
        role = message.role
        content = message.content

        if role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
        elif role == "system":
            prompt = f"{content}\n" + prompt

    prompt += "<|assistant|>\n"
    return prompt

# --- 3. FastAPI 应用和 API 接口定义 ---
app = FastAPI(
    title="DeepSeek-R1-Distill-Qwen-1.5B OpenAI Compatible API",
    description="提供与OpenAI Chat Completions API兼容的接口，由ModelScope上的DeepSeek模型驱动。",
    version="0.1.0",
)

# OpenAI API 请求体模型
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model name, e.g., 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature.")
    top_p: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling probability.")
    max_tokens: Optional[int] = Field(512, ge=1, description="Maximum number of tokens to generate.")
    stream: Optional[bool] = Field(False, description="Whether to stream partial progress.")
    # 可以添加更多OpenAI兼容的参数，如 stop, presence_penalty, frequency_penalty等

# OpenAI API 响应体模型 - Message
class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str

# OpenAI API 响应体模型 - Choice
class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: Optional[str] = None # "stop", "length", "content_filter"

# OpenAI API 响应体模型 - Usage
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# OpenAI API 响应体模型 - Main
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server is not ready.")

    if request.stream:
        # TODO: Implement streaming response. This example only supports non-streaming for now.
        raise HTTPException(status_code=501, detail="Streaming is not yet implemented.")

    # 1. 构建模型输入 prompt
    full_prompt = build_prompt_from_openai_messages(request.messages)

    # 2. 编码输入
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    prompt_token_count = inputs["input_ids"].shape[1] # 计算prompt token数量

    # 3. 生成输出
    output_ids = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=request.temperature,
        top_p=request.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, # 确保 pad_token_id 与 eos_token_id 一致
    )

    # 4. 解码并清理响应
    # output_ids[0] for batch size 1
    # output_ids[0][inputs["input_ids"].shape[-1]:] for only newly generated tokens
    generated_text_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_response = tokenizer.decode(generated_text_ids, skip_special_tokens=True).strip()

    # 清理输出：移除模型可能意外生成的特殊标记
    response_content = raw_response.replace("<|user|>", "").replace("<|assistant|>", "").strip()

    # 5. 计算生成 token 数量
    completion_token_count = generated_text_ids.shape[0]
    total_tokens = prompt_token_count + completion_token_count

    # 6. 构建 OpenAI 兼容的响应
    choice_message = ChoiceMessage(content=response_content)
    choice = Choice(index=0, message=choice_message, finish_reason="stop") # 假设是正常停止

    usage = Usage(
        prompt_tokens=prompt_token_count,
        completion_tokens=completion_token_count,
        total_tokens=total_tokens
    )

    response_data = ChatCompletionResponse(
        model=request.model,
        choices=[choice],
        usage=usage
    )

    return response_data

# --- 4. 启动服务 ---
if __name__ == "__main__":
    # 启动前检查模型是否加载成功
    if model is None or tokenizer is None:
        print("模型未能成功加载，API 服务将不会启动。请检查模型加载错误。")
        exit(1) # 如果模型加载失败，则退出
    
    # 使用 uvicorn 启动 FastAPI 应用
    # --host 0.0.0.0 允许从外部访问
    # --port 8000 是常见的API端口
    print("\nDeepSeek OpenAI 兼容 API 服务正在启动...")
    print("请访问 http://127.0.0.1:8000/docs 查看API文档。")
    uvicorn.run(app, host="0.0.0.0", port=8000)