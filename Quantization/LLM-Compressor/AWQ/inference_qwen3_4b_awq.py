from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载量化后的模型和tokenizer
model_path = "./Qwen3-4B-AWQ"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,  # 使用 dtype 替代 torch_dtype
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# --- 重要：设置 pad_token ---
# 如果 tokenizer 没有 pad_token，则使用 eos_token 作为 pad_token
# 这对于防止注意力掩码警告通常是必要的
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Warning: pad_token was None. Set pad_token to eos_token: '{tokenizer.eos_token}'")

# 定义一个辅助函数来生成回复
def generate_response(instruction, input_text=""):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction + "\n" + input_text if input_text else instruction}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 编码输入文本，明确使用 tokenizer 的 pad_token
    # padding=False 表示不进行填充，这通常适用于单次推理，可以避免该警告
    # 但如果后续处理需要，也可以设置 padding=True, pad_to_max_length=True 等
    # 此处保持 padding=False，因为 generate 通常不需要对输入进行填充
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    input_ids = inputs["input_ids"].to(model.device)
    # 注意：即使 padding=False, tokenizer 也可能生成 attention_mask，但通常为空或不必要
    # 在调用 generate 时，v1.0+ 的 transformers 通常能自动处理

    # 为了更加明确和安全，可以传递 attention_mask
    attention_mask = inputs.get("attention_mask").to(model.device) if "attention_mask" in inputs else None

    # 模型生成
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask, # 传递 attention_mask
        max_new_tokens=256,
        temperature=0.6,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id, # 使用 pad_token_id 而不是 eos_token_id
        # eos_token_id=tokenizer.eos_token_id, # 如果需要，也可以显式指定结束符
    )

    # 解码生成的输出，跳过输入部分
    # 注意：decode 时要跳过输入的 token 长度
    generated_ids = outputs[0][len(input_ids[0]):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

# --- 推理测试 ---
print("--- Qwen3-4B AWQ 量化模型推理测试 ---")

instruction1 = "请解释一下量子计算是什么。"
response1 = generate_response(instruction1)
print(f"Q: {instruction1}\nA: {response1}\n")

instruction2 = "用Python写一个快速排序的函数。"
response2 = generate_response(instruction2)
print(f"Q: {instruction2}\nA: {response2}\n")

instruction3 = "对以下输入进行总结：大型语言模型（LLM）是基于深度学习的 AI 程序，能够理解、生成和翻译自然语言。"
input_text3 = ""
response3 = generate_response(instruction3, input_text3)
print(f"Q: {instruction3}\nA: {response3}\n")

print("--- 测试完成 ---")
