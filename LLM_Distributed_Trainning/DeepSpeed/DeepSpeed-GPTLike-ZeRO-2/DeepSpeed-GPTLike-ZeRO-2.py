#!/usr/bin/env python3
# train_deepspeed_stage2.py

"""
DeepSpeed Stage-2 简化 GPT-like 训练脚本（初学者友好版）
特点：
- 支持 ZeRO Stage-2（优化器状态和梯度分片）
- 自动读取 ds_config_stage2.json 中 train_micro_batch_size_per_gpu
- 自动使用 DeepSpeed launcher 传入的 local_rank，无需手动传入
- 中文注释，简洁明了，适合教学使用
"""

import os
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from datasets import load_dataset

# DeepSpeed
try:
    import deepspeed
except Exception:
    deepspeed = None

# 日志配置
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# 数据加载
# -----------------------------
def prepare_data():
    """加载 WikiText 数据集并返回非空文本列表"""
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
        self.block_size = block_size
        if block_size > 512:
            raise ValueError(f"block_size ({block_size}) 超过最大序列长度 512")
        logger.info("编码文本...")
        all_ids = []
        for t in texts:
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
        return block[:-1], block[1:]

# -----------------------------
# 模型定义
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
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
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, int(d_model * 4), dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTLike(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, d_model, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
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
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx):
        B, L = idx.size()
        if L > self.block_size:
            idx = idx[:, :self.block_size]
            L = self.block_size

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :L, :].expand(B, -1, -1)
        x = self.drop(tok_emb + pos_emb)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

# -----------------------------
# 主训练流程
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="简化 GPT-like 模型训练（DeepSpeed Stage-2）")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="本地 DataLoader 批次大小")
    parser.add_argument("--block_size", type=int, default=256, help="序列块大小（<=512）")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=12, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=768, help="模型维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")
    parser.add_argument("--ds_config", type=str, default="ds_config.json", help="DeepSpeed 配置文件路径")
    parser.add_argument("--local_rank", type=int, default=0, help="由 DeepSpeed launcher 自动传入")
    args = parser.parse_args()

    if args.block_size > 512:
        raise ValueError(f"block_size ({args.block_size}) 超过最大序列长度 512")
    if args.d_model % args.n_head != 0:
        raise ValueError(f"d_model ({args.d_model}) 必须能被 n_head ({args.n_head}) 整除")

    # 读取 DeepSpeed 配置文件中的 train_micro_batch_size_per_gpu
    ds_batch = None
    if args.ds_config and os.path.isfile(args.ds_config):
        with open(args.ds_config, 'r') as f:
            ds_conf = json.load(f)
        ds_batch = ds_conf.get('train_micro_batch_size_per_gpu')
        if ds_batch is not None:
            ds_batch = int(ds_batch)
    dataloader_batch_size = ds_batch if ds_batch is not None else args.batch_size
    logger.info(f"DataLoader batch_size={dataloader_batch_size}")

    if deepspeed is None:
        logger.error("未检测到 deepspeed，请先安装: pip install deepspeed")
        return

    # 初始化 DeepSpeed 分布式
    try:
        deepspeed.init_distributed()
    except Exception:
        pass

    world_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = world_rank == 0
    if is_main_process:
        logger.info(f"DeepSpeed 环境：rank={world_rank}, world_size={world_size}")

    # 设置设备
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if is_main_process:
        logger.info(f"使用设备: {device}")

    # 加载数据和分词器
    texts = prepare_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    if is_main_process:
        logger.info(f"BERT 分词器词汇大小: {vocab_size}")

    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size)
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=dataloader_batch_size, sampler=sampler,
                            shuffle=(sampler is None), drop_last=True, num_workers=4)

    # 初始化模型和优化器
    model = GPTLike(vocab_size, args.block_size, args.n_layer, args.n_head, args.d_model, args.dropout)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # DeepSpeed 初始化（ZeRO Stage-2 在 ds_config 中指定）
    engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        config=args.ds_config
    )

    Path("checkpoints").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # 训练循环
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        engine.train()
        total_loss, num_batches = 0.0, 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            logits = engine(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            engine.backward(loss)
            engine.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0 and is_main_process:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}, Loss={loss.item():.4f}")

        if num_batches > 0 and is_main_process:
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Avg Loss={avg_loss:.4f}")

        # 保存检查点
        engine.save_checkpoint("checkpoints", tag=f"epoch{epoch+1}")
        if is_main_process:
            logger.info(f"已保存检查点: checkpoints/epoch{epoch+1}")

    # 保存最终模型权重
    if is_main_process:
        torch.save(engine.module.state_dict(), "models/final_model.pth")
        logger.info("最终模型已保存至 models/final_model.pth")

    # 优雅关闭 DeepSpeed 和分布式进程组
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            if is_main_process:
                logger.info("已成功销毁 torch.distributed 进程组")
    except Exception as e:
        logger.warning(f"销毁分布式进程组时出错: {e}")

if __name__ == "__main__":
    main()
