#!/usr/bin/env python3
"""
极简版BPE语言模型训练
功能:
- 训练BPE分词器
- 小型Transformer语言模型（固定正弦/余弦位置编码）
- 训练、保存模型及生成文本示例
"""
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
import matplotlib.pyplot as plt

# -----------------------------
# 0. 设置随机种子确保可复现
# -----------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# -----------------------------
# 1. 数据准备与预处理
# -----------------------------
def prepare_data():
    """加载并预处理 WikiText 数据集"""
    print("正在加载 WikiText 数据集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [example["text"] for example in dataset if len(example["text"].strip()) > 0]
    
    # 保存文本到文件，用于分词器训练
    os.makedirs("data", exist_ok=True)
    with open("data/training_text.txt", "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    
    print(f"已准备 {len(texts)} 条文本数据")
    return texts

# -----------------------------
# 2. 训练 BPE 分词器
# -----------------------------
def train_tokenizer(vocab_size=30000):
    """训练 BPE 分词器"""
    print("正在训练 BPE 分词器...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True
    )
    tokenizer.train(["data/training_text.txt"], trainer)
    tokenizer.save("data/bpe_tokenizer.json")
    print(f"分词器训练完成，词汇表大小: {vocab_size}")
    return tokenizer

# -----------------------------
# 3. 文本数据集
# -----------------------------
class TextDataset(Dataset):
    """文本数据集，将文本编码为块序列"""
    def __init__(self, texts, tokenizer, block_size=64):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []
        
        all_token_ids = []
        for text in texts:
            if text.strip():
                encoded = tokenizer.encode(text)
                all_token_ids.extend(encoded.ids)
        
        # 切块，每块长度为 block_size
        for i in range(0, len(all_token_ids) - block_size, block_size):
            self.data.append(all_token_ids[i:i+block_size])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx][:-1]  # 输入序列
        y = self.data[idx][1:]   # 目标序列（右移一位）
        return torch.tensor(x), torch.tensor(y)

# -----------------------------
# 4. 简化 Transformer 模型（固定正弦/余弦位置编码）
# -----------------------------
class SimpleTransformer(nn.Module):
    """极简 Transformer 语言模型"""
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=6, max_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 固定正弦/余弦位置编码
        self.register_buffer("pos_embedding", self._get_sinusoidal_embeddings(max_len, d_model))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def _get_sinusoidal_embeddings(self, seq_len, d_model):
        """生成固定正弦/余弦位置编码 (1, seq_len, d_model)"""
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        """前向传播"""
        x_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding[:, :x.size(1), :]
        x = x_emb + pos_emb
        x = self.transformer(x)
        logits = self.output_layer(x)
        return logits

# -----------------------------
# 5. 模型训练函数
# -----------------------------
def train_model(model, dataloader, epochs=3, lr=0.001):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    print("开始训练模型...")
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {avg_loss:.4f}")
        
        avg_epoch_loss = total_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
    
    # 绘制损失曲线
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/training_loss.png")
    print("训练完成! 损失曲线已保存至 data/training_loss.png")
    
    return model

# -----------------------------
# 6. 文本生成函数
# -----------------------------
def generate_text(model, tokenizer, prompt="The meaning of life is", max_length=20):
    """使用训练好的模型生成文本"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_tensor)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
    
    generated_ids = input_tensor.squeeze().cpu().tolist()
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

# -----------------------------
# 7. 主执行函数
# -----------------------------
def main():
    # 准备数据
    texts = prepare_data()
    
    # 训练分词器
    tokenizer = train_tokenizer(vocab_size=3000)
    tokenizer = Tokenizer.from_file("data/bpe_tokenizer.json")
    
    # 数据集与数据加载器
    dataset = TextDataset(texts, tokenizer, block_size=64)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建模型
    vocab_size = tokenizer.get_vocab_size()
    model = SimpleTransformer(vocab_size=vocab_size, max_len=64)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    model = train_model(model, dataloader, epochs=3)
    
    # 保存模型
    torch.save(model.state_dict(), "data/simple_transformer.pth")
    print("模型已保存至 data/simple_transformer.pth")
    
    # 文本生成示例
    print("\n--- 文本生成示例 ---")
    examples = ["The meaning of life is", "Machine learning is", "In the future,"]
    for prompt in examples:
        generated = generate_text(model, tokenizer, prompt=prompt, max_length=20)
        print(f"输入: '{prompt}'")
        print(f"生成: '{generated}'\n")

if __name__ == "__main__":
    main()
