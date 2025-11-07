from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
import torch

model_path = "/home/marion/Pretrained_Models/Qwen3-4B"
dataset_name = "llm-wizard/alpaca-gpt4-data-zh"

# 1. 加载 tokenizer
# GPTQModel 在量化时可以自动加载 tokenizer，但手动加载也无妨
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. 加载原始模型用于量化
# 注意：使用 GPTQModel.load 并传入 quantize_config 来准备量化
# 这会返回一个 GPTQModel 实例，该实例可以执行量化
quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    # dtype=torch.float16 # 可以指定加载原始模型的 dtype，如果需要
)

# 使用 GPTQModel.load 加载原始模型，并传入量化配置
# trust_remote_code=True 对于 Qwen 模型是必要的
model = GPTQModel.load(
    model_path, # 原始模型路径
    quantize_config=quantize_config,
    trust_remote_code=True
)

# 3. 加载数据集并构建样本 (与原脚本类似，但 GPTQModel 期望文本列表)
dataset = load_dataset(dataset_name, split="train[:200]")
calibration_dataset = [] # GPTQModel 通常期望一个字符串列表或字典列表
for i in range(min(128, len(dataset))): # 确保不超过数据集大小
    text = dataset[i]["instruction"] + "\n" + dataset[i]["input"] + "\n" + dataset[i]["output"]
    calibration_dataset.append(text)

# 4. 执行量化
# 调用 model.quantize 方法，传入校准数据集
# batch_size 可以根据显存情况调整
# use_triton=False 对应 GPTQModel 中的某些后端选项，这里默认即可
model.quantize(calibration_dataset, batch_size=1)

# 5. 保存量化模型
output_dir = "./Qwen3-4B-GPTQ"
model.save(output_dir) # GPTQModel 使用 save 方法
# tokenizer 也需要保存，虽然 GPTQModel.load 有时可以自动加载，但手动保存更保险
tokenizer.save_pretrained(output_dir)

print("Qwen3-4B使用gptqmodel量化完成，结果保存在：", output_dir)
