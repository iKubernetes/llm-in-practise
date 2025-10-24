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
from torch.utils.data import DataLoader
import torch

# 1. 加载数据集并替换占位符
dataset = load_dataset("modelscope/self-cognition", split="train")
model_name = "马哥教育AI小助手" # 设置模型名称
model_author = "马哥教育AI团队" # 设置开发者名称

def replace_placeholders(example):
    # 替换 response 中的占位符 {{NAME}} 和 {{AUTHOR}}
    example["response"] = example["response"].replace("{{NAME}}", model_name).replace("{{AUTHOR}}", model_author)
    return example
dataset = dataset.map(replace_placeholders)
#print("步骤 1：数据集加载并替换占位符完成，示例：", dataset[0])

# 2. 转换为 Chat Messages 格式
def to_chat_messages(example):
    # 创建 system 消息，定义助手身份
    system_message = {"role": "system", "content": "你是一个有帮助的智能助手，由马哥教育AI团队训练，名为马哥教育AI小助手，旨在提供准确且友好的回答。"}
    user_message = {"role": "user", "content": example["query"]}
    assistant_message = {"role": "assistant", "content": example["response"]}
    example["messages"] = [system_message, user_message, assistant_message]
    return {"messages": example["messages"]}
chat_messages_dataset = dataset.map(to_chat_messages, remove_columns=["query", "response", "tag"])
#print("步骤 2：转换为 Chat Messages 格式完成，示例：", chat_messages_dataset[0])

# 3. 渲染为 ChatML 模板
def to_chatml_template(example):
    # 获取 messages 列表并构造 ChatML 格式
    messages = example["messages"]
    chatml_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        chatml_text += f"<|im_start|>{role}\n{content} <|im_end|>\n"
    example["chatml_text"] = chatml_text.strip()
    return example
chatml_dataset = chat_messages_dataset.map(to_chatml_template) 
#print("步骤 3：渲染为 ChatML 风格模板格式完成，示例：", chatml_dataset[0]["chatml_text"])

# 4. 使用 Qwen3-8B Tokenizer 分词
#model_checkpoint = "Qwen/Qwen3-8B" # Qwen3-8B的checkpoint，需确认实际路径
model_checkpoint = "/home/marion/Pretrained_Models/Qwen3-8B" # Qwen3-8B的checkpoint，需确认实际路径
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token # 确保有 pad_token

def tokenize_for_sft(examples):
    # 对 chatml_text 进行分词
    tokenized = tokenizer(
        examples["chatml_text"],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    # 为 SFT 生成 labels，忽略 system 和 user 部分
    input_ids = tokenized["input_ids"]
    labels = input_ids.clone()
    for i in range(len(input_ids)):
        # 寻找 Assistant 回复的起始位置
        assistant_start = (input_ids[i] == tokenizer.encode("<|im_start|>assistant")[0]).nonzero(as_tuple=True)[0]
        if assistant_start.numel() > 0:
            # 忽略 System 和 User 部分 (将起始位置之前的部分设置为 -100)
            labels[i, :assistant_start[0]] = -100 
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = chatml_dataset.map(tokenize_for_sft, batched=True, remove_columns=["messages", "chatml_text"])
tokenized_dataset.set_format("torch")

# 设置 DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
train_dataloader = DataLoader(tokenized_dataset, batch_size=4, collate_fn=data_collator, shuffle=True)

# 5. 配置QLoRA并应用到模型
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    device_map="auto", #在分布式训练中，应该让accelerator或Trainer来管理设备分配
    low_cpu_mem_usage=True,
    dtype=torch.bfloat16, 
    use_cache=False,
    # 使用从transformers导入的BitsAndBytesConfig
    quantization_config=BitsAndBytesConfig( 
        load_in_4bit=True, # 保持量化设置在配置对象中
        # 调整为 bfloat16
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_quant_type="nf4",
        # 推荐启用双重量化以保持精度
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
model.print_trainable_parameters() # 打印可训练参数量

# 6. 设置训练参数并运行微调
output_dir = "./finetuned/qwen3-8b-qlora"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=3, 
    learning_rate=5e-5, 
    logging_steps=10,
    save_steps=10,
    eval_strategy="no",
    save_total_limit=3,
    load_best_model_at_end=False,
    push_to_hub=False,
    bf16=True, # 启用 bfloat16 训练
    optim="paged_adamw_8bit", # 推荐使用 8bit 优化器，与 QLoRA 兼容
    gradient_checkpointing=True, # 启用梯度检查点，节省显存

    report_to=[],  # 禁用wandb等记录器，避免分布式环境下的冲突
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
print("开始QLoRA微调...")
#trainer.train()
#print("微调完成！")

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
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"模型和Tokenizer已保存至 {output_dir}")
