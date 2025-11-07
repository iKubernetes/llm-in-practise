"""
Qwen3-4B GPTQ 量化模型推理脚本（使用 apply_chat_template）
适配最新 gptqmodel API：from_quantized
"""

from gptqmodel import GPTQModel
from transformers import AutoTokenizer
import torch

# 配置
MODEL_PATH = "./Qwen3-4B-GPTQ"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# 1. 加载 Tokenizer 和量化模型
print(f"正在加载量化模型: {MODEL_PATH}")
print(f"使用设备: {DEVICE}")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=True
)

model = GPTQModel.from_quantized(
    MODEL_PATH,
    device=DEVICE,
    trust_remote_code=True
)

model.eval()
print("模型加载完成！\n")

# 2. 对话生成函数
def chat_generate(
    messages,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
    stream=False
):
    """
    使用 apply_chat_template 进行对话生成
    """
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if stream:
        print("回复: ", end="", flush=True)
        generated_tokens = []
        # 注意：gptqmodel 的 stream=True 返回的是逐步增长的 input_ids
        for output_ids in model.generate(**generate_kwargs, stream=True):
            new_token_id = output_ids[0, -1].item() if output_ids.shape[1] > inputs["input_ids"].shape[1] else None
            if new_token_id is not None:
                token_str = tokenizer.decode(new_token_id, skip_special_tokens=True)
                print(token_str, end="", flush=True)
                generated_tokens.append(new_token_id)
        print()
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        with torch.no_grad():
            output_ids = model.generate(**generate_kwargs)
        response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

    return response.strip()


# 3. 测试
if __name__ == "__main__":
    print("=" * 60)
    print("     Qwen3-4B GPTQ 量化模型对话测试（apply_chat_template）")
    print("=" * 60)

    # 测试 1：单轮问答
    print("\n【测试 1：单轮问答】")
    messages1 = [
        {"role": "user", "content": "请用中文介绍一下北京的故宫，控制在100字以内。"}
    ]
    reply1 = chat_generate(messages1, max_new_tokens=128)
    print(f"用户: {messages1[0]['content']}")
    print(f"模型: {reply1}\n")

    # 测试 2：多轮对话
    print("【测试 2：多轮对话】")
    messages2 = [
        {"role": "user", "content": "我计划去北京旅游三天。"},
        {"role": "assistant", "content": "很好！三天时间可以深度游览北京的核心景点。我来帮你规划一下。"},
        {"role": "user", "content": "第一天我想去故宫和天安门，第二天呢？"}
    ]
    reply2 = chat_generate(messages2, max_new_tokens=256)
    print(f"用户: {messages2[-1]['content']}")
    print(f"模型: {reply2}\n")

    print("所有测试完成！")
