from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 加载数据集并替换占位符
dataset = load_dataset("modelscope/self-cognition", split="train")
model_name = "马哥教育AI小助手"
model_author = "马哥教育AI团队"

def replace_placeholders(example):
    example["response"] = example["response"].replace("{{NAME}}", model_name).replace("{{AUTHOR}}", model_author)
    return example

dataset = dataset.map(replace_placeholders)
#print("步骤 1：数据集加载并替换占位符完成")

# 2. 转换为 Chat Messages 格式
def to_chat_messages(example):
    system_message = {"role": "system", "content": "你是一个有帮助的智能助手，由马哥教育AI团队训练，名为马哥教育AI小助手，旨在提供准确且友好的回答。"}
    user_message = {"role": "user", "content": example["query"]}
    assistant_message = {"role": "assistant", "content": example["response"]}
    example["messages"] = [system_message, user_message, assistant_message]
    return {"messages": example["messages"]}

chat_messages_dataset = dataset.map(to_chat_messages, remove_columns=["query", "response", "tag"])
#print("步骤 2：转换为 Chat Messages 格式完成")

# 3. 渲染为 ChatML 模板
def to_chatml_template(example):
    messages = example["messages"]
    chatml_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        chatml_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    example["chatml_text"] = chatml_text.strip()
    return example

chatml_dataset = chat_messages_dataset.map(to_chatml_template)
#print("步骤 3：渲染为 ChatML 风格模板格式完成")

# 4. 加载 Tokenizer 和分词
model_checkpoint = "/home/marion/Pretrained_Models/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

# 确保有 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_for_sft(examples):
    # 对 chatml_text 进行分词
    tokenized = tokenizer(
        examples["chatml_text"],
        padding=True,  # 改为动态padding
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    
    # 为 SFT 生成 labels
    input_ids = tokenized["input_ids"]
    labels = input_ids.clone()
    
    # 找到所有特殊token的位置
    for i in range(len(input_ids)):
        # 找到 assistant 开始的位置
        assistant_token = tokenizer.encode("<|im_start|>assistant")[0]
        assistant_positions = (input_ids[i] == assistant_token).nonzero(as_tuple=True)[0]
        
        if len(assistant_positions) > 0:
            assistant_start = assistant_positions[0]
            # 将 assistant 之前的所有token的label设为 -100
            labels[i, :assistant_start] = -100
            
            # 确保 assistant 内容部分有正确的labels
            # 找到 assistant 结束的位置
            end_token = tokenizer.encode("<|im_end|>")[0]
            end_positions = (input_ids[i] == end_token).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                # 找到 assistant 对应的结束位置
                for end_pos in end_positions:
                    if end_pos > assistant_start:
                        # 将结束符之后的label也设为-100
                        labels[i, end_pos+1:] = -100
                        break
    
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = chatml_dataset.map(tokenize_for_sft, batched=True, remove_columns=["messages", "chatml_text"])
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 设置 DataCollator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False,
    pad_to_multiple_of=8  # 优化GPU内存使用
)

# 5. 加载基础模型并配置LoRA（不使用量化）
print("加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False,  # 禁用缓存以支持梯度检查点
)

# 启用梯度检查点（在应用LoRA之前）
model.gradient_checkpointing_enable()

print(f"模型设备: {next(model.parameters()).device}")

# 配置 LoRA 参数 - 使用更全面的目标模块
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # 增加秩以获得更好性能
    lora_alpha=32,
    lora_dropout=0.05,  # 降低dropout
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    bias="none",
    modules_to_save=None,  # 确保没有其他模块被设置为可训练
)

# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)

# 打印可训练参数信息
model.print_trainable_parameters()

# 检查模型参数梯度设置
print("检查模型参数梯度...")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数数量: {trainable_params}")

if trainable_params == 0:
    raise ValueError("没有可训练参数！请检查LoRA配置。")

# 6. 设置训练参数并运行微调
output_dir = "./finetuned/qwen3-8b-lora"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # 减小批次大小
    gradient_accumulation_steps=4,  # 增加梯度累积
    num_train_epochs=3,  # 减少训练轮数
    learning_rate=1e-4,  # 调整学习率
    logging_steps=10,
    save_steps=100,
    eval_strategy="no",
    save_total_limit=2,
    load_best_model_at_end=False,
    push_to_hub=False,
    bf16=True,
    optim="adamw_torch",  # 使用标准优化器
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    report_to=[],  # 禁用wandb
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,  # 确保不删除必要列
    label_names=["labels"],  # 明确指定label列
)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    #tokenizer=tokenizer,
)

print("开始LoRA微调...")
try:
    # 开始训练
    train_result = trainer.train()
    print("微调完成！")
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
except Exception as e:
    print(f"训练过程中出现错误: {e}")
    # 保存当前状态
    trainer.save_model(output_dir + "_interrupted")
    raise

# 7. 保存微调后的模型
print("保存模型...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"LoRA适配器和Tokenizer已保存至 {output_dir}")

# 可选：保存完整的合并模型
#print("合并LoRA权重到基础模型...")
#merged_model = model.merge_and_unload()
#merged_output_dir = output_dir + "_merged"
#merged_model.save_pretrained(merged_output_dir)
#tokenizer.save_pretrained(merged_output_dir)
#print(f"完整合并模型已保存至 {merged_output_dir}")
