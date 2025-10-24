import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ==================== 分布式环境配置 ====================
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

print(f"分布式训练信息:")
print(f"  Local Rank: {local_rank}")
print(f"  World Size: {world_size}")
print(f"  可用 GPU 数量: {torch.cuda.device_count()}")

# 设置当前设备
torch.cuda.set_device(local_rank)

# ==================== 数据预处理 ====================
# 1. 加载数据集并替换占位符
dataset = load_dataset("modelscope/self-cognition", split="train")
model_name = "马哥教育AI小助手"
model_author = "马哥教育AI团队"

def replace_placeholders(example):
    example["response"] = example["response"].replace("{{NAME}}", model_name).replace("{{AUTHOR}}", model_author)
    return example

dataset = dataset.map(replace_placeholders)

# 2. 转换为 Chat Messages 格式
def to_chat_messages(example):
    system_message = {"role": "system", "content": "你是一个有帮助的智能助手，由马哥教育AI团队训练，名为马哥教育AI小助手，旨在提供准确且友好的回答。"}
    user_message = {"role": "user", "content": example["query"]}
    assistant_message = {"role": "assistant", "content": example["response"]}
    example["messages"] = [system_message, user_message, assistant_message]
    return {"messages": example["messages"]}

chat_messages_dataset = dataset.map(to_chat_messages, remove_columns=["query", "response", "tag"])

# 3. 渲染为 ChatML 模板
def to_chatml_template(example):
    messages = example["messages"]
    chatml_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        chatml_text += f"<|im_start|>{role}\n{content} <|im_end|>\n"
    example["chatml_text"] = chatml_text.strip()
    return example

chatml_dataset = chat_messages_dataset.map(to_chatml_template)

# ==================== 分词处理 ====================
#model_checkpoint = "Qwen/Qwen3-14B" # Qwen3-14B
model_checkpoint = "/home/marion/Pretrained_Models/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_for_sft(examples):
    tokenized = tokenizer(
        examples["chatml_text"],
        padding="max_length",
        max_length=512,
        truncation=True,
    )
    
    input_ids = torch.tensor(tokenized["input_ids"])
    labels = input_ids.clone()
    
    # 查找 assistant_start_token_id
    assistant_start_token_id = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[0]

    for i in range(len(input_ids)):
        assistant_start = (input_ids[i] == assistant_start_token_id).nonzero(as_tuple=True)[0]
        if assistant_start.numel() > 0:
            labels[i, :assistant_start[0]] = -100 
    
    tokenized["labels"] = labels.tolist()
    return tokenized

tokenized_dataset = chatml_dataset.map(tokenize_for_sft, batched=True, remove_columns=["messages", "chatml_text"])

# ==================== 模型配置 ====================
# 数据整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 模型加载 - 分布式优化
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    device_map=None if world_size > 1 else "auto",  # 多GPU时让Trainer管理设备分配
    low_cpu_mem_usage=True,
    #dtype=torch.bfloat16,   #若使用load_in_4bit=True，则模型dtype由quantization_config决定
    use_cache=False,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
)

# 准备模型支持k-bit训练
model = prepare_model_for_kbit_training(model)

# 配置 LoRA 参数
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)

# 应用 LoRA 到量化模型
model = get_peft_model(model, lora_config)

# 只在主进程打印可训练参数
if local_rank == 0:
    model.print_trainable_parameters()

# ==================== 训练配置 ====================
output_dir = "./finetuned/qwen3-14b-qlora-dist"

training_args = TrainingArguments(
    output_dir=output_dir,
    
    # 批次大小配置
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    
    # 训练周期和学习率
    num_train_epochs=3,
    learning_rate=5e-5,
    
    # 日志和保存配置
    logging_steps=10,
    save_steps=10,
    eval_strategy="no",
    save_total_limit=3,
    load_best_model_at_end=False,
    
    # 优化器配置
    optim="paged_adamw_8bit",
    
    # 分布式训练配置
    ddp_find_unused_parameters=False,  # LoRA训练必须设置为False
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    local_rank=local_rank,
    
    # 精度和内存优化
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # 其他配置
    push_to_hub=False,
    report_to=[],  # 禁用wandb等记录器，避免分布式环境下的冲突
    
    # 分布式通信配置
    ddp_backend="nccl",
    ddp_timeout=1800,  # 增加超时时间

    # 非标准字段（如 chatml_text），有时会被 Trainer 自动删除，导致 collator 出错
    remove_unused_columns=False,
)

# ==================== 训练器初始化 ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ==================== 开始训练 ====================
#if local_rank == 0:
if trainer.is_world_process_zero():
    print("开始QLoRA微调...")
    print("训练配置:")
    print(f"  总GPU数量: {world_size}")
    print(f"  每设备批次大小: {training_args.per_device_train_batch_size}")
    print(f"  梯度累积步数: {training_args.gradient_accumulation_steps}")
    print(f"  有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size}")

# 开始训练
trainer.train()

# ==================== 模型保存 ====================
# 使用Trainer的保存方法，自动处理分布式环境
trainer.save_model(output_dir)

# 只在主进程保存tokenizer和打印完成信息
if trainer.is_world_process_zero():
    tokenizer.save_pretrained(output_dir)
    print("微调完成！")
    print(f"模型和Tokenizer已保存至{output_dir}")

# ==================== 清理资源 ====================
# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
