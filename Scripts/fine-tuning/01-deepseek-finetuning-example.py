import torch
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# --- 1. 配置参数 ---
MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
DATASET_ID = 'ms_hackathon_23_agent_train_dev'
OUTPUT_DIR = "./finetuned_model"
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512 # 根据模型和数据调整最大序列长度
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"] # 适用于Qwen模型

# --- 2. 数据集和模型加载 ---
print(f"正在下载模型：{MODEL_ID}...")
model_dir = snapshot_download(MODEL_ID)

# 修正：明确指定模型设备到 cuda:0
# 如果您的机器有多个GPU，通常默认使用cuda:0。
# 如果您想使用其他GPU，例如cuda:1，可以改为 'cuda:1'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"将模型加载到设备: {device}")
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 为Qwen tokenizer设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"正在加载数据集：{DATASET_ID}...")
dataset = MsDataset.load(DATASET_ID, split='train') # 假设使用 'train' 切分
print(f"数据集大小: {len(dataset)}")
print(f"数据集第一个样本: {dataset[0]}") # 打印这里，仔细观察它的结构！

# --- 3. 数据预处理 ---
def preprocess_function(examples):
    texts = []
    
    if 'query' in examples and 'response' in examples:
        for q, r in zip(examples['query'], examples['response']):
            texts.append(f"User: {q}\nAgent: {r}{tokenizer.eos_token}")
    elif 'instruction' in examples and 'output' in examples:
        for instr, out in zip(examples['instruction'], examples['output']):
            texts.append(f"Instruction: {instr}\nOutput: {out}{tokenizer.eos_token}")
    elif 'text' in examples:
        for t in examples['text']:
            texts.append(f"{t}{tokenizer.eos_token}")
    else:
        print("警告：未找到预期的 'query', 'response', 'instruction', 'output' 或 'text' 字段。")
        print("将尝试拼接每个样本的所有字符串类型的值。请务必检查数据集结构并根据需要调整 preprocess_function。")

        batch_size = len(next(iter(examples.values()))) if examples else 0
        
        if batch_size == 0:
            return tokenizer([], max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length")

        for i in range(batch_size):
            combined_text = ""
            for col_name in examples:
                value = examples[col_name][i]
                if isinstance(value, str):
                    combined_text += f"{col_name}: {value}\n"
                elif isinstance(value, (int, float)):
                    combined_text += f"{col_name}: {str(value)}\n"
            texts.append(combined_text.strip() + tokenizer.eos_token)


    tokenized_inputs = tokenizer(
        texts,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length"
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

print("正在预处理数据集...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    # remove_columns=dataset.column_names # 移除原始列以节省内存，如果不需要原始列
)
print(f"预处理后的数据集第一个样本: {tokenized_dataset[0]}")

# --- 4. 配置 PEFT (Parameter-Efficient Fine-Tuning) ---
print("正在配置 LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model.enable_input_require_grads() # 启用梯度计算
peft_model = get_peft_model(model, lora_config)
print("可训练参数量：")
peft_model.print_trainable_parameters()

# --- 5. 训练配置与启动 ---
print("正在配置训练参数并启动训练...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=50,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    fp16=True, # 开启混合精度训练 (如果您的GPU支持)
    # bf16=True, # 如果您的GPU支持bfloat16，可以开启
    # data_collator 可以帮助将批次数据移动到正确的设备
    # 默认的 DataCollatorForLanguageModeling 应该能处理
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# --- 6. 保存微调后的模型 ---
print(f"微调完成！正在保存模型到 {OUTPUT_DIR}_final...")
peft_model.save_pretrained(f"{OUTPUT_DIR}_final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}_final")

print("模型保存成功。")