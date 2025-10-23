import torch
import numpy as np
from datasets import load_dataset             # 用于加载 Hugging Face 数据集
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate                             # Hugging Face 官方评估库，用于加载指标


# 2. 数据准备与预处理
# 定义使用的模型名称和超参数
MODEL_NAME = "bert-base-uncased" # 选择一个 BERT 风格模型
MAX_LENGTH = 128                 # 序列最大长度
TRAIN_SAMPLE_PERCENT = 5         # 仅使用 1% 的训练数据进行快速演示
EVAL_SAMPLE_PERCENT = 5          # 仅使用 1% 的测试数据进行评估

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 加载 IMDB 数据集，并按百分比截取
print(f"--- 正在加载数据集 (仅使用 {TRAIN_SAMPLE_PERCENT}% 训练集和 {EVAL_SAMPLE_PERCENT}% 测试集) ---")
# 标签列名为 'label'，文本列名为 'text'
raw_train_datasets = load_dataset("imdb", split=f"train[:{TRAIN_SAMPLE_PERCENT}%]")
raw_eval_datasets = load_dataset("imdb", split=f"test[:{EVAL_SAMPLE_PERCENT}%]")

# 定义预处理（分词）函数
def tokenize_function(examples):
    """将文本转换为模型所需的 token ID 和注意力掩码"""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

# 对数据集应用预处理（并行处理）
tokenized_train_datasets = raw_train_datasets.map(tokenize_function, batched=True)
tokenized_eval_datasets = raw_eval_datasets.map(tokenize_function, batched=True)

# 将原始标签列 'label' 重命名为 'labels'，这是 Trainer 默认识别的标签列名
tokenized_train_datasets = tokenized_train_datasets.rename_column("label", "labels")
tokenized_eval_datasets = tokenized_eval_datasets.rename_column("label", "labels")

# 移除原始文本列，只保留模型训练需要的 columns: input_ids, token_type_ids, attention_mask, labels
tokenized_train_datasets = tokenized_train_datasets.remove_columns(["text"])
tokenized_eval_datasets = tokenized_eval_datasets.remove_columns(["text"])

# 最终训练和评估数据集
train_dataset = tokenized_train_datasets
eval_dataset = tokenized_eval_datasets

# 3. 模型加载与指标函数定义 (Evaluate 库功能演示)
# 3.1 加载模型：使用 AutoModelForSequenceClassification 进行二分类任务
NUM_LABELS = 2
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# 3.2 加载评估指标：使用 evaluate 库加载准确率 (accuracy) 指标
metric = evaluate.load("accuracy")

def compute_metrics(p):
    """
    Trainer 在评估时会调用此函数。
    输入 p 是一个 EvalPrediction 对象 (包含 predictions 和 label_ids)
    """
    # 提取模型的 logits 或其他预测结果
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # 转换为最终的类别预测 ID
    predictions = np.argmax(logits, axis=1)
    
    # 使用 evaluate 库计算指标
    return metric.compute(predictions=predictions, references=p.label_ids)

# 4. 实例化 Trainer 并训练 (Trainer 库核心功能演示)
# 4.1 定义训练参数 (TrainingArguments)
OUTPUT_DIR = "./results_trainer_demo"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,                   # 检查点和输出的目录
    num_train_epochs=1,                      # 训练周期数，设置为 1 快速完成
    per_device_train_batch_size=8,           # 每个设备上的训练批次大小
    per_device_eval_batch_size=8,            # 每个设备上的评估批次大小
    warmup_steps=100,                        # 学习率预热步数
    weight_decay=0.01,                       # 权重衰减
    logging_dir='./logs_trainer_demo',       # TensorBoard 日志目录
    logging_steps=50,
    eval_strategy="epoch",                   # 评估策略: 每个 epoch 结束时进行评估
    save_strategy="epoch",                   # 保存策略: 每个 epoch 结束时保存模型
    load_best_model_at_end=True,             # 训练结束后加载基于 metric_for_best_model 的最佳模型
    metric_for_best_model="accuracy",        # 定义衡量最佳模型的指标
    report_to="none",                        # 关闭外部报告工具以保持简洁
)

# 4.2 实例化 Trainer
trainer = Trainer(
    model=model,                             # 传入加载的模型
    args=training_args,                      # 传入训练参数
    train_dataset=train_dataset,             # 传入处理后的训练集
    eval_dataset=eval_dataset,               # 传入处理后的评估集
    tokenizer=tokenizer,                     # 传入分词器（用于数据整理和保存）
    compute_metrics=compute_metrics,         # 传入指标计算函数
)

# 4.3 开始训练
print("\n" + "="*50)
print("--- 启动模型训练 (Trainer.train()) ---")
print("="*50)
train_result = trainer.train()

# 5. 评估和结果打印
# 5.1 记录训练统计信息（如总步数、总时间）
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state() # 保存 Trainer 状态

# 5.2 使用 Trainer.evaluate() 进行最终评估
print("\n" + "="*50)
print("--- 启动模型评估 (Trainer.evaluate()) ---")
print("="*50)
eval_results = trainer.evaluate(eval_dataset=eval_dataset)

# 5.3 打印评估结果，包括由 compute_metrics 计算的准确率
print("\n最终评估结果:")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# 6. 模型保存（可选）
# 保存最终模型和分词器
FINAL_SAVE_PATH = f"{OUTPUT_DIR}/final_model"
print(f"\n--- 保存最终模型到: {FINAL_SAVE_PATH} ---")
trainer.save_model(FINAL_SAVE_PATH)
tokenizer.save_pretrained(FINAL_SAVE_PATH)

print("\n--- 任务流程已完成 ---")
