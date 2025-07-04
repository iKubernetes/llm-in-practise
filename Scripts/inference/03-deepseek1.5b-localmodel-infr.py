from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 模型本地路径
local_model_path = "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B"

# 加载 tokenizer 和模型（注意 trust_remote_code 保留）
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

# 用户输入
user_input = "请简单介绍一下量子计算的基本原理。"
prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 推理生成
generate_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
)

# 解码输出
output = tokenizer.decode(generate_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

print("模型回答：")
print(output)