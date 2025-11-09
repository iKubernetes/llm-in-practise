from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
# 改为从 awq 子模块导入
from llmcompressor.modifiers.awq import AWQModifier

import torch
import os

model_path = "/home/marion/temp/LLaMA-Factory/merged/Deepseek-R1-0528-Qwen3-MageduAI"
dataset_name = "llm-wizard/alpaca-gpt4-data-zh"

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. 定义量化配置（使用 AWQModifier）
recipe = []
recipe.append(
    AWQModifier(
        targets="Linear",        # 作用在模型的线性层上
        scheme="W4A16",     # 非对称压缩，若要使用对称压缩，则修改为“W4A16_SYM”
        ignore=["lm_head"],      # 跳过输出层
        # 如果需要 mapping，可加参数 mappings=[…]
    )
)

# 3. 加载校准数据
raw_dataset = load_dataset(dataset_name, split="train[:128]")
calibration_texts = []
for i in range(len(raw_dataset)):
    instr = raw_dataset[i].get("instruction", "") or raw_dataset[i].get("prompt", "") or ""
    inp = raw_dataset[i].get("input", "") or ""
    outp = raw_dataset[i].get("output", "") or raw_dataset[i].get("completion", "") or raw_dataset[i].get("text", "")
    text = (instr + "\n" + inp + "\n" + outp).strip()
    if text == "":
        text = " ".join(str(v) for v in raw_dataset[i].values())
    calibration_texts.append({"text": text})

calibration_dataset = Dataset.from_list(calibration_texts)

# 4. 执行量化
output_dir = "./DeepSeek-R1-0528-Qwen3-8B-AWQ"
os.makedirs(output_dir, exist_ok=True)

oneshot(
    model=model_path,
    recipe=recipe,
    tokenizer=tokenizer,
    dataset=calibration_dataset,
    output_dir=output_dir,
    max_seq_length=2048,
    num_calibration_samples=len(calibration_dataset),
    #quantization_format="awq",           # 强制导出标准 AWQ
    #export_to_marlin=True,               # 导出 Marlin kernel（vLLM 最快）
    #save_compressed=True,
    #model_config=config,
)

# 5. 保存 tokenizer
tokenizer.save_pretrained(output_dir)

print("DeepSeek-R1-Qwen3‑8B-MageEduAI使用llm‑compressor AWQ量化完成，结果保存在：", output_dir)
