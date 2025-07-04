import os
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from modelscope.hub.snapshot_download import snapshot_download

# ========= 可选：在脚本内设置缓存目录（或使用 MODELSCOPE_CACHE 环境变量） =========
# os.environ['MODELSCOPE_CACHE'] = '/your/custom/path'

# 模型参数
model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
model_dir = snapshot_download(model_id)

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ========= 多轮对话上下文 =========
history = []

# ========= 多轮对话输入函数 =========
def build_prompt(history):
    prompt = ""
    for turn in history:
        prompt += f"<|user|>\n{turn['user']}\n<|assistant|>\n{turn['assistant']}\n"
    return prompt

# ========= 推理函数 =========
def chat(user_input):
    global history

    # 构造完整 prompt（不加当前用户输入）
    past_prompt = build_prompt(history)
    new_prompt = f"{past_prompt}<|user|>\n{user_input}\n<|assistant|>\n"

    # Tokenize 输入
    inputs = tokenizer(new_prompt, return_tensors="pt").to(model.device)

    # 模型生成
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 解码新生成的 assistant 回复部分
    output = tokenizer.decode(
        generate_ids[0][inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # 更新对话历史
    history.append({
        "user": user_input,
        "assistant": output
    })

    return output

# ========= 控制台对话入口 =========
if __name__ == "__main__":
    print("欢迎使用 DeepSeek-Qwen 多轮对话 Demo，输入 'exit' 退出。")
    while True:
        user_input = input("用户：")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = chat(user_input)
        print(f"模型：{reply}\n")