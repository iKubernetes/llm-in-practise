import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 路径设置
# 模型ID或本地保存的模型路径
# base_model_path = "Qwen3/Qwen3-8B" 
base_model_path = "/home/marion/Pretrained_Models/Qwen3-8B" 
# LoRA/QLoRA微调输出路径
lora_model_path = "./finetuned/qwen3-8b-lora"

# 2. 模型加载 
print("正在加载Tokenizer与LoRA/QLoRA权重...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 加载LoRA/QLoRA权重
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

# 3. 生成函数
def chat(query, history=None, system_prompt=None, max_new_tokens=256):
    """
    构建ChatML格式输入并推理输出。
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages += history
    messages.append({"role": "user", "content": query})

    # 构造ChatML输入
    chatml_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        chatml_text += f"<|im_start|>{role}\n{content} <|im_end|>\n"
    chatml_text += "<|im_start|>assistant\n"

    inputs = tokenizer(chatml_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip()


# 4. 系统提示词
system_prompt = (
    "你是一个有帮助的智能助手，由马哥教育AI团队训练，名为马哥教育AI小助手，旨在提供准确且友好的回答。"
)

# 5. 测试问题
questions = [
    "你是？",
    "请用三句话介绍你自己。"
]

# 6. 执行推理
print("\n============== 开始测试(MageEdu AI助手) ==============\n")
history = []
for i, q in enumerate(questions, 1):
    print(f"[问题 {i}] {q}")
    answer = chat(q, history=history, system_prompt=system_prompt)
    print(f"[回答 {i}] {answer}\n")
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": answer})

print("============== 测试结束(MageEdu AI助手) ==============")

