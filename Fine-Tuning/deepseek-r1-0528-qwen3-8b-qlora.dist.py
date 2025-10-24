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
    system_message = {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。今天是2025年10月23日。"}
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
        chatml_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    example["chatml_text"] = chatml_text.strip()
    return example

chatml_dataset = chat_messages_dataset.map(to_chatml_template)

# ==================== 分词处理 ====================
model_checkpoint = "/home/marion/Pretrained_Models/DeepSeek-R1-0528-Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
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
    
    # 查找 assistant_start_token_id，添加安全检查
    try:
        assistant_start_token_id = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[0]
    except IndexError:
        print("警告: '<|im_start|>assistant' 未在 tokenizer 中找到，使用默认 assistant 标记")
        assistant_start_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]

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

# 模型加载 - 分布式优化，修复 rope_scaling 和 use_cache 问题
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    device_map=None if world_size > 1 else "auto",
    low_cpu_mem_usage=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    ),
    rope_scaling=None,  # 禁用 rope_scaling 以避免 'attn_factor' 警告
    use_cache=False,    # 明确设置 use_cache=False 以兼容 gradient checkpointing
    trust_remote_code=True
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
output_dir = "./finetuned/deepseek-r1-0528-qwen3-8b-qlora-dist"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=10,
    eval_strategy="no",
    save_total_limit=3,
    load_best_model_at_end=False,
    optim="paged_adamw_8bit",
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    local_rank=local_rank,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    push_to_hub=False,
    report_to=[],
    ddp_backend="nccl",
    ddp_timeout=1800,
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
trainer.save_model(output_dir)

# 只在主进程保存tokenizer和打印完成信息
if trainer.is_world_process_zero():
    tokenizer.save_pretrained(output_dir)
    print("微调完成！")
    print(f"模型和Tokenizer已保存至{output_dir}")

# ==================== 清理资源 ====================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
