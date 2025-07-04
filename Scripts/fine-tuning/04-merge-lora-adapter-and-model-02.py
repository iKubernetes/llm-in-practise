import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # 仅在使用 LoRA 进行微调时才需要
from modelscope.hub.snapshot_download import snapshot_download
import os

# --- 配置 ---
# 原始模型 ID (用于下载基础模型)
ORIGINAL_MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'

# 您微调后模型权重的路径
# 如果您使用了 LoRA 脚本，这就是您的适配器路径 (例如: "./finetuned_model_custom_id")
# 如果您使用了全量微调脚本，这就是您的完整模型路径 (例如: "./finetuned_model_full_id")
# 请务必根据您实际使用的微调方法来调整此路径。
FINETUNED_MODEL_PATH = "./finetuned_model_custom_id" # <--- 重要：请调整此路径！

# 合并后模型将保存的路径
MERGED_MODEL_PATH = "./merged_finetuned_deepseek_vllm"

# --- 1. 加载原始基础模型和分词器 ---
print(f"正在下载并加载原始基础模型: {ORIGINAL_MODEL_ID}...")
original_model_dir = snapshot_download(ORIGINAL_MODEL_ID)

# 以与训练时相同的数据类型 (推荐 bfloat16) 加载基础模型
# device_map="auto" 有助于在有多个 GPU 时将模型层分布到不同 GPU 上
base_model = AutoModelForCausalLM.from_pretrained(
    original_model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(original_model_dir)

# --- 2. 加载并合并权重 (根据微调方法条件执行) ---
# 检查是否存在 LoRA 适配器文件，以判断微调方式
if "peft_model.safetensors" in os.listdir(FINETUNED_MODEL_PATH) or \
   "adapter_model.safetensors" in os.listdir(FINETUNED_MODEL_PATH):
    # 如果检测到 LoRA 适配器，则执行此块
    print(f"检测到 LoRA 适配器。正在从以下路径加载并合并适配器: {FINETUNED_MODEL_PATH}...")
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
    model = model.merge_and_unload() # 将 LoRA 权重合并到基础模型中
else:
    # 如果是全量微调模型，则直接加载
    print(f"检测到全量微调模型。正在直接从以下路径加载: {FINETUNED_MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # 分词器可能也随全量模型一起保存，但我们已经加载了基础分词器。
    # 如果分词器被修改过，请确保也从 FINETUNED_MODEL_PATH 加载。
    # tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)


# --- 3. 保存合并后的模型和分词器 ---
print(f"正在保存合并后的模型到: {MERGED_MODEL_PATH}...")
os.makedirs(MERGED_MODEL_PATH, exist_ok=True) # 如果目录不存在则创建
model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print(f"合并后的模型已成功保存到: {MERGED_MODEL_PATH}")