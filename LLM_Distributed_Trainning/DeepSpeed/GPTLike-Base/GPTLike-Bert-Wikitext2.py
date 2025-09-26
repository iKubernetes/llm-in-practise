#!/usr/bin/env python3
"""
简化的 GPT-like 模型训练脚本，专注于快速验证
- 使用 BERT 分词器和 wikitext-2-raw-v1 数据集
- 固定正弦/余弦位置编码
- 每 50 批次打印平均损失，每 epoch 保存检查点
- 最终模型保存到 models 目录
- 修复超长序列警告，通过截断确保序列长度不超过 512
- 支持命令行配置模型层数、头数、维度和 dropout
启动示例: python train.py --epochs 3 --batch_size 16 --n_layer 8 --n_head 16 --d_model 1024 --dropout 0.2
"""
import os
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

# 配置日志
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# 数据加载与分词
# -----------------------------
def prepare_data():
    """加载 wikitext-2-raw-v1 数据集的训练集，返回非空文本列表"""
    logger.info("加载 WikiText 数据集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex["text"] for ex in dataset if ex.get("text") and ex["text"].strip()]
    logger.info(f"加载 {len(texts)} 条非空文本")
    return texts

# -----------------------------
# 数据集
# -----------------------------
class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=256):
        """初始化数据集，按块组织编码后的 token，确保序列长度不超过 512"""
        self.block_size = block_size
        if block_size > 512:
            raise ValueError(f"block_size ({block_size}) 超过 BERT 最大序列长度 512")
        logger.info("编码文本...")
        all_ids = []
        for t in texts:
            # 使用 max_length 和 truncation 限制序列长度
            token_ids = tokenizer.encode(t, add_special_tokens=False, max_length=512, truncation=True)
            all_ids.extend(token_ids)
        
        total_len = (len(all_ids) // block_size) * block_size
        if total_len == 0:
            raise ValueError(f"数据不足以构成 block_size={block_size} 的块")
        
        arr = torch.tensor(all_ids[:total_len], dtype=torch.long)
        self.data = arr.view(-1, block_size)
        logger.info(f"总 token 数={len(all_ids)}, 块数={len(self.data)}")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        block = self.data[idx]
        return block[:-1], block[1:]  # 输入和目标（偏移一位）

# -----------------------------
# 模型定义
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        """初始化因果自注意力模块"""
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """前向传播，使用上三角掩码实现因果注意力"""
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        """初始化前馈网络模块"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        """初始化 Transformer 块，包含自注意力和前馈网络"""
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, int(d_model * 4), dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 残差连接：注意力
        x = x + self.mlp(self.ln2(x))   # 残差连接：前馈
        return x

class GPTLike(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, d_model, dropout=0.1):
        """初始化 GPT-like 模型，使用固定正弦/余弦位置编码"""
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # 共享输入和输出嵌入权重
        self.block_size = block_size

        # 固定正弦/余弦位置编码
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pe.unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重：线性层和嵌入层使用 Xavier 初始化，LayerNorm 初始化为标准值"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx):
        """前向传播：输入 token 索引，输出 logits"""
        B, L = idx.size()
        if L > self.block_size:
            idx = idx[:, :self.block_size]
            L = self.block_size

        tok_emb = self.tok_emb(idx)  # token 嵌入
        pos_emb = self.pos_emb[:, :L, :].expand(B, -1, -1)  # 位置编码
        x = self.drop(tok_emb + pos_emb)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

# -----------------------------
# 主训练流程
# -----------------------------
def main():
    """主函数：解析参数、设置环境、训练模型并保存检查点"""
    import argparse
    parser = argparse.ArgumentParser(description="简化的 GPT-like 模型训练")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--block_size", type=int, default=256, help="序列块大小（不得超过 512）")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=12, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=768, help="模型维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")
    args = parser.parse_args()

    # 验证 block_size 不超过 512
    if args.block_size > 512:
        raise ValueError(f"block_size ({args.block_size}) 超过 BERT 最大序列长度 512")

    # 验证 d_model 可被 n_head 整除
    if args.d_model % args.n_head != 0:
        raise ValueError(f"d_model ({args.d_model}) 必须能被 n_head ({args.n_head}) 整除")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载数据和分词器
    texts = prepare_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    logger.info(f"BERT 分词器词汇大小: {vocab_size}")

    # 创建数据集和加载器
    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)

    # 初始化模型，使用命令行参数配置
    model = GPTLike(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 创建保存目录
    Path("checkpoints").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss, num_batches = 0.0, 0
        batch_losses = []

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = sum(batch_losses[-50:]) / min(len(batch_losses), 50)
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}, Avg Loss (last 50): {avg_loss:.4f}")

        # Epoch 结束，打印平均损失并保存检查点
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

    # 保存最终模型
    torch.save(model.state_dict(), "models/final_model.pth")
    logger.info("最终模型已保存至 models/final_model.pth")

if __name__ == "__main__":
    main()
