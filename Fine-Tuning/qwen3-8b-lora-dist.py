import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. 分布式环境配置
# LOCAL_RANK：当前进程在本节点的 GPU 编号（由 torchrun 自动传入）
# WORLD_SIZE：全局 GPU 数量
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

print(f"分布式训练信息：")
print(f"  Local Rank: {local_rank}")
print(f"  World Size: {world_size}")
print(f"  可用 GPU 数量: {torch.cuda.device_count()}")

# 设置当前 GPU 设备（必须在加载模型前设置，否则 CUDA context 冲突）
torch.cuda.set_device(local_rank)

# 2. 数据预处理
# 加载数据集并替换占位符
dataset = load_dataset("modelscope/self-cognition", split="train")
model_name = "马哥教育AI小助手"
model_author = "马哥教育AI团队"

def replace_placeholders(example):
    """替换模板占位符"""
    example["response"] = example["response"].replace("{{NAME}}", model_name).replace("{{AUTHOR}}", model_author)
    return example

dataset = dataset.map(replace_placeholders)

# 转换为 Chat 格式
def to_chat_messages(example):
    """构造 system + user + assistant 三段对话结构"""
    system_message = {
        "role": "system",
        "content": "你是一个有帮助的智能助手，由马哥教育AI团队训练，名为马哥教育AI小助手，旨在提供准确且友好的回答。"
    }
    user_message = {"role": "user", "content": example["query"]}
    assistant_message = {"role": "assistant", "content": example["response"]}
    example["messages"] = [system_message, user_message, assistant_message]
    return {"messages": example["messages"]}

chat_messages_dataset = dataset.map(to_chat_messages, remove_columns=["query", "response", "tag"])

# 渲染为 ChatML 模板格式（<|im_start|> + role + 内容）
def to_chatml_template(example):
    """将 messages 转换为 ChatML 格式文本"""
    messages = example["messages"]
    chatml_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        chatml_text += f"<|im_start|>{role}\n{content} <|im_end|>\n"
    example["chatml_text"] = chatml_text.strip()
    return example

chatml_dataset = chat_messages_dataset.map(to_chatml_template)

# 3. 分词处理
model_checkpoint = "/home/marion/Pretrained_Models/Qwen3-8B"  # 预训练模型路径
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token  # 确保存在 pad_token（部分模型未定义）

def tokenize_for_sft(examples):
    """对 ChatML 文本进行分词，并屏蔽掉 user 段的标签"""
    tokenized = tokenizer(
        examples["chatml_text"],
        padding="max_length",
        max_length=512,
        truncation=True,
    )

    input_ids = tokenized["input_ids"]
    labels = []
    assistant_start_token_id = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[0]

    # 屏蔽掉 assistant 之前的内容（即 user/system）
    for ids in input_ids:
        label = ids.copy()
        if assistant_start_token_id in ids:
            idx = ids.index(assistant_start_token_id)
            label[:idx] = [-100] * idx  # -100 表示忽略梯度计算
        labels.append(label)

    tokenized["labels"] = labels
    return tokenized

# 进行分词映射
tokenized_dataset = chatml_dataset.map(
    tokenize_for_sft,
    batched=True,
    remove_columns=["messages", "chatml_text"]
)

# 4. LoRA 配置
# 数据整理器（语言建模场景禁用 MLM）
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 加载预训练模型（不使用量化）
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    device_map=None if world_size > 1 else "auto",  # 多GPU交由 Trainer 管理
    dtype=torch.bfloat16,                     # 推荐 bfloat16（或 float16）
    low_cpu_mem_usage=True,
    use_cache=False                                # 训练时禁用 cache 以节省显存
)

# 配置LoRA低秩适配器
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # 自回归语言建模任务
    r=8,                            # LoRA rank
    lora_alpha=16,                  # 缩放系数
    lora_dropout=0.1,               # dropout比例
    target_modules=["q_proj", "v_proj"],  # 应用到注意力的 Q/V 投影层
    bias="none"                     # 不修改 bias
)

# 应用LoRA到原始模型
model = get_peft_model(model, lora_config)

# 只在主进程打印可训练参数统计信息
if local_rank == 0:
    model.print_trainable_parameters()

# 5. 训练参数配置
output_dir = "./finetuned/qwen3-8b-lora-dist"

training_args = TrainingArguments(
    output_dir=output_dir,

    # 训练批次配置
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,

    # 学习率与训练周期
    num_train_epochs=3,
    learning_rate=5e-5,

    # 日志与保存
    logging_steps=10,
    save_steps=50,
    eval_strategy="no",
    save_total_limit=2,

    # 优化器配置
    optim="adamw_torch",  # 非量化可直接使用标准AdamW
    weight_decay=0.01,

    # 混合精度与显存优化
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # 分布式配置
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,  # LoRA必须设为False
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    local_rank=local_rank,

    # 其他配置
    remove_unused_columns=False,
    report_to=[],  # 关闭wandb等日志
)

# 6. Trainer 初始化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 7. 开始训练
if trainer.is_world_process_zero():
    print("开始 LoRA 微调...")
    print("训练配置:")
    print(f"  总GPU数量: {world_size}")
    print(f"  每设备批次大小: {training_args.per_device_train_batch_size}")
    print(f"  梯度累积步数: {training_args.gradient_accumulation_steps}")
    print(f"  有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size}")

trainer.train()

# ==================== 模型保存 ====================
# Trainer.save_model() 会自动处理分布式环境同步
trainer.save_model(output_dir)

# 主进程额外保存分词器
if trainer.is_world_process_zero():
    tokenizer.save_pretrained(output_dir)
    print("微调完成！")
    print(f"模型和Tokenizer已保存至 {output_dir}")

# ==================== 清理资源 ====================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
