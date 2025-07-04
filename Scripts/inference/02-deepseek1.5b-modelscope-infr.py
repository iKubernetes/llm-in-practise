from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型 ID（来自魔搭社区）
model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'  # 确保该模型已在 ModelScope 上发布

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

# 推理输入
user_input = "请简单介绍一下量子计算的基本原理。"
prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成输出
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