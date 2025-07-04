import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from modelscope.hub.snapshot_download import snapshot_download
import os

# --- 配置路径 ---
# 原始模型ID，用于下载基模型
ORIGINAL_MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
# 微调后LoRA适配器保存的路径 (对应您脚本中的 OUTPUT_DIR_final)
FINETUNED_ADAPTER_PATH = "./finetuned_model_final"
# 合并后模型的保存路径
MERGED_MODEL_PATH = "./merged_finetuned_model"

# --- 1. 下载原始模型和分词器 ---
print(f"正在下载原始模型：{ORIGINAL_MODEL_ID}...")
original_model_dir = snapshot_download(ORIGINAL_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    original_model_dir,
    torch_dtype=torch.bfloat16, # 使用与训练时相同的dtype
    device_map="auto" # 合并时可以使用auto，推理时再指定单卡
)
tokenizer = AutoTokenizer.from_pretrained(original_model_dir)

# --- 2. 加载 LoRA 适配器 ---
print(f"正在加载微调后的 LoRA 适配器：{FINETUNED_ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, FINETUNED_ADAPTER_PATH)

# --- 3. 合并 LoRA 适配器到基模型 ---
print("正在合并 LoRA 适配器到基模型...")
# `merge_and_unload()` 方法会返回一个合并了LoRA权重的完整模型
model = model.merge_and_unload()

# --- 4. 保存合并后的模型和分词器 ---
print(f"正在保存合并后的模型到：{MERGED_MODEL_PATH}...")
os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print(f"合并后的模型已成功保存到：{MERGED_MODEL_PATH}")