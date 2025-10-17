# 0. 确保必要的库已安装
# pip install transformers datasets torch numpy evaluate

# 1. 导入所需库
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding  # 导入用于动态填充的数据收集器
)
from datasets import load_dataset
import numpy as np
from evaluate import load
import torch

# 2. 设置训练时要使用的GPU，请注意要对照实际的硬件环境进行修改
# 设置环境变量，只允许脚本看到并使用系统中的 GPU 0 和 GPU 2
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,2" 
print(f"可见GPU数: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 3. 加载数据集
# 加载常用的IMDb电影评论数据集（包含'train'和'test'拆分）
dataset = load_dataset("imdb")
print("原始数据集结构:")
print(dataset)

# 4. 加载预训练模型和分词器
model_name = "bert-base-uncased"  # 选择BERT基础模型（小写）
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 5. 数据预处理与标记化 (使用动态填充的最佳实践)
def tokenize_function(examples):
    # 对文本进行分词，注意：这里我们只进行截断，不进行填充，
    # 填充工作交给 DataCollatorWithPadding 在批次内完成，以提高效率。
    return tokenizer(
        examples["text"],
        truncation=True,        # 截断过长的序列
        max_length=512          # 设置最大序列长度
    )

# 应用分词函数到整个数据集
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    desc="正在对数据集进行分词处理"
)

# 移除原始文本列和多余的列，只保留模型需要的 'input_ids', 'attention_mask', 'labels'
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
# 将 'label' 列重命名为 'labels' 以匹配 PyTorch 模型的默认期望
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 设置为 PyTorch 张量格式，方便后续处理
tokenized_datasets.set_format("torch")

# 6. 准备模型
# 加载用于序列分类的BERT模型，指定标签数量（IMDb是二分类任务：正面/负面）
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 7. 定义评估指标函数
def compute_metrics(eval_pred):
    """
    负责在评估阶段计算准确率等指标。
    """
    # 加载准确率评估指标
    metric = load("accuracy")
    logits, labels = eval_pred  # eval_pred 是一个元组 (模型输出的logits, 真实标签)
    
    # 将 logits 转换为预测标签 (取最大值的索引)
    predictions = np.argmax(logits, axis=-1)
    
    # 使用 evaluate 库计算准确率
    return metric.compute(predictions=predictions, references=labels)

# 8. 配置数据收集器
# 显式使用 DataCollatorWithPadding，它会根据批次内最长序列动态填充，优化训练速度。
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 9. 配置训练参数
training_args = TrainingArguments(
    output_dir="./bert_imdb_results",     # 输出目录，用于保存模型和结果
    
    # 核心超参数
    learning_rate=2e-5,                   # 微调BERT的常用学习率
    per_device_train_batch_size=8,        # 每个设备的训练批次大小
    per_device_eval_batch_size=8,         # 每个设备的评估批次大小
    num_train_epochs=3,                   # 训练总轮数
    weight_decay=0.01,                    # 权重衰减，用于防止过拟合
    fp16=True,                            # 开启混合精度训练，加速GPU训练
    
    # 评估和保存策略
    eval_strategy="epoch",          # 每个epoch结束后进行评估
    save_strategy="epoch",                # 每个epoch结束后保存模型检查点
    load_best_model_at_end=True,          # 训练结束后加载在评估集上表现最好的模型
    metric_for_best_model="accuracy",     # 以准确率作为评判最佳模型的标准
    
    # 日志和报告
    logging_dir="./logs",                 # TensorBoard 日志目录
    logging_steps=500,                    # 每500步记录一次日志
    # push_to_hub=True,                   # 如果想推送到 Hugging Face Hub 可以取消注释
)

# 10. 初始化Trainer
# 注意：这里我们使用 "test" 集作为评估集，但理想情况下应该从训练集中划分出专门的验证集。
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # 使用测试集作为评估集（为简化示例）
    tokenizer=tokenizer,
    data_collator=data_collator,          # 传递数据收集器
    compute_metrics=compute_metrics,      # 传递评估函数
)

# 11. 开始训练！
print("--- 训练开始 ---")
train_results = trainer.train()

# 12. 评估最终模型
# 使用 trainer.evaluate() 对模型进行最终评估
eval_results = trainer.evaluate()
print(f"最终评估结果: {eval_results}")

# 13. 保存微调后的最佳模型和分词器
# Trainer 已经自动保存了最佳模型，这里使用 trainer.save_model() 
# 将最佳模型和分词器统一保存到指定路径，代码更简洁。
save_path = "./fine_tuned_bert_imdb"
trainer.save_model(save_path)
print(f"最佳模型和分词器已保存至: {save_path}")
