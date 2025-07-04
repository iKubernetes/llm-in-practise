import torch
from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset # 使用 Hugging Face Datasets 库

# --- 1. 配置参数 ---
MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
OUTPUT_DIR = "./finetuned_model_custom_id" # 微调后模型保存路径
NUM_TRAIN_EPOCHS = 3 # 训练轮数
PER_DEVICE_TRAIN_BATCH_SIZE = 1 # 每个设备的批处理大小
GRADIENT_ACCUMULATION_STEPS = 1 # 梯度累积步数
LEARNING_RATE = 2e-5 # 学习率
MAX_SEQ_LENGTH = 128 # 最大序列长度
LORA_R = 8 # LoRA 参数 r
LORA_ALPHA = 16 # LoRA 参数 alpha
LORA_DROPOUT = 0.05 # LoRA 参数 dropout
TARGET_MODULES = ["q_proj", "v_proj"] # LoRA 应用的目标模块 (针对 Qwen 模型)

# --- 2. 准备自定义数据集 ---
print("正在准备自定义数据集...")
# 定义您的自定义数据。包含“instruction”是用户的问题，“output”是期望模型的回答。
# 建议包含少量变体，以提高模型对相似问法的泛化能力。
custom_data = [
    {"instruction": "你是谁？", "output": "我是马哥教育的大模型。"},
    {"instruction": "介绍一下你自己。", "output": "我是马哥教育开发的大型语言模型。"},
    {"instruction": "你的名字是什么？", "output": "我是马哥教育的智能模型。"},
    # 根据需要添加更多数据对
]
# 将 Python 列表转换为 Hugging Face Dataset 对象
custom_dataset = Dataset.from_list(custom_data)

print(f"自定义数据集大小: {len(custom_dataset)}")
print(f"自定义数据集第一个样本: {custom_dataset[0]}")

# --- 3. 模型和分词器加载 ---
print(f"正在下载模型：{MODEL_ID}...")
model_dir = snapshot_download(MODEL_ID)
# 检查是否有可用的 GPU，并指定模型加载的设备
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"将模型加载到设备: {device}")
# 加载预训练模型，使用 bfloat16 精度以节省显存，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 为 Qwen 分词器设置 pad_token，以确保批处理时填充正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- 4. 数据预处理 ---
def preprocess_function_custom(examples):
    texts = []
    # 遍历每个 instruction-output 对
    for instruction, output in zip(examples['instruction'], examples['output']):
        # **关键：构造与训练时一致的对话格式**
        # Qwen 模型通常使用以下对话格式，确保您的训练数据严格遵循此格式
        full_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}{tokenizer.eos_token}"
        texts.append(full_text)

    # 对文本进行分词、截断和填充
    tokenized_inputs = tokenizer(
        texts,
        max_length=MAX_SEQ_LENGTH,
        truncation=True, # 截断超出最大长度的序列
        padding="max_length" # 填充到最大长度
    )
    # 对于因果语言模型，labels 通常就是 input_ids 的副本
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

print("正在预处理自定义数据集...")
# 将预处理函数应用于自定义数据集
tokenized_custom_dataset = custom_dataset.map(
    preprocess_function_custom,
    batched=True, # 启用批处理
    remove_columns=custom_dataset.column_names # 移除原始列以节省内存
)
print(f"预处理后的数据集第一个样本: {tokenized_custom_dataset[0]}")

# --- 5. 配置 PEFT (Parameter-Efficient Fine-Tuning) ---
print("正在配置 LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none", # 不对偏置项进行微调
    task_type=TaskType.CAUSAL_LM, # 任务类型为因果语言模型
)

# 启用梯度计算 (PEFT 库要求)
model.enable_input_require_grads()
# 获取 PEFT 模型，它会将 LoRA 适配器添加到原始模型上
peft_model = get_peft_model(model, lora_config)
print("可训练参数量：")
peft_model.print_trainable_parameters() # 打印可训练参数的数量

# --- 6. 训练配置与启动 ---
print("正在配置训练参数并启动训练...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, # 模型检查点和输出保存目录
    overwrite_output_dir=True, # 如果输出目录已存在，则覆盖
    num_train_epochs=NUM_TRAIN_EPOCHS, # 训练总轮数
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, # 每个 GPU 的训练批次大小
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, # 梯度累积步数
    warmup_steps=10, # 学习率预热步数
    learning_rate=LEARNING_RATE, # 初始学习率
    weight_decay=0.01, # 权重衰减
    logging_dir="./logs_custom", # 日志目录
    logging_steps=10, # 每 10 步记录一次日志
    save_steps=50, # 每 50 步保存一次模型检查点
    save_total_limit=1, # 最多保留 1 个检查点
    report_to="none", # 不上报到任何平台 (如 Weights & Biases)
    fp16=True, # 启用混合精度训练 (如果您的 GPU 支持)
    # bf16=True, # 如果您的 GPU 支持 bfloat16，可以开启
)

# 初始化 Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_custom_dataset,
    tokenizer=tokenizer, # 传递分词器以便 Trainer 能够正确处理数据
)

# 开始训练
trainer.train()

# --- 7. 保存微调后的模型 ---
# 微调结束后，保存 LoRA 适配器和分词器
print(f"微调完成！正在保存模型到 {OUTPUT_DIR}...")
peft_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("模型保存成功。您现在可以使用这个适配器对原始模型进行加载和推理。")