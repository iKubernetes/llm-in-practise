# ==========================================
# 基于 llm-compressor 的 GPTQ 量化脚本（适配 llmcompressor==0.8.1）
# ==========================================

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
import torch
import os

model_path = "/home/marion/Pretrained_Models/Qwen3-4B/"
dataset_name = "llm-wizard/alpaca-gpt4-data-zh"

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. 定义量化配置
recipe = []
recipe.append(
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )
)

# 3. 加载校准数据
raw_dataset = load_dataset(dataset_name, split="train[:128]")

# 把 list[str] 转换成 datasets.Dataset
# llm-compressor 需要 Dataset 对象
calibration_texts = []
for i in range(len(raw_dataset)):
    instr = raw_dataset[i].get("instruction", "") or raw_dataset[i].get("prompt", "") or ""
    inp = raw_dataset[i].get("input", "") or ""
    outp = raw_dataset[i].get("output", "") or raw_dataset[i].get("completion", "") or raw_dataset[i].get("text", "")
    text = (instr + "\n" + inp + "\n" + outp).strip()
    if text == "":
        text = " ".join(str(v) for v in raw_dataset[i].values())
    calibration_texts.append({"text": text})

# 转换成 Dataset 对象
calibration_dataset = Dataset.from_list(calibration_texts)

# 4. 执行量化
output_dir = "./Qwen3-4B-GPTQ"
os.makedirs(output_dir, exist_ok=True)

oneshot(
    model=model_path,
    recipe=recipe,
    tokenizer=tokenizer,
    dataset=calibration_dataset,
    output_dir=output_dir,
    max_seq_length=2048,
    num_calibration_samples=len(calibration_dataset),
    #quantization_format="gptq",     # 默认量化格式为compress-tensors，该选项可强制为gptq
    #export_to_marlin=False,        # GPTQ 不支持 marlin
    #save_compressed=True,
    #model_config=config,
)

# 5. 保存 tokenizer
tokenizer.save_pretrained(output_dir)

print("Qwen3-4B 使用 llm-compressor GPTQ 量化完成，结果保存在：", output_dir)
#print("若保存为gptq量化格式，则结果可直接用于 vLLM/AutoGPTQ/ExLlamaV2 加载")
