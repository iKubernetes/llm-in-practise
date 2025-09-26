#!/usr/bin/env python3
# train_deepspeed_stage2_wikitext2.py
"""
基于 DeepSpeed ZeRO Stage-2 的简化 GPT-like 训练脚本（教学友好版）
- 使用 WikiText-2 数据集
- 模型规格相较原脚本增大（默认：12 层，d_model=1024，16 头）
- 中文注释，尽量简洁，便于教学
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
from datasets import load_dataset
from transformers import GPT2TokenizerFast

# DeepSpeed
try:
    import deepspeed
except Exception:
    deepspeed = None

# 日志配置（便于调试和跟踪训练过程）
logging.basicConfig(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# 数据加载（WikiText-2）
# -----------------------------
def prepare_data():
    """加载 wikitext-2 raw 数据集的 train split，返回非空文本列表"""
    logger.info("加载 WikiText-2 数据集（train split）...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex["text"] for ex in ds if ex.get("text") and ex["text"].strip()]
    logger.info(f"加载到 {len(texts)} 条非空文本")
    return texts

# -----------------------------
# 数据集（按 token 串切块）
# -----------------------------
class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=512):
        self.block_size = block_size
        logger.info("对文本进行分词并拼接成 token 流...")
        all_ids = []
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=False, max_length=block_size, truncation=True)
            all_ids.extend(ids)
        total_len = (len(all_ids) // block_size) * block_size
        if total_len == 0:
            raise ValueError("数据不足以构成任何一个 block，请检查数据集")
        arr = torch.tensor(all_ids[:total_len], dtype=torch.long)
        # 每个样本为 block_size 的 token 序列，输入为[:-1]，标签为[1:]
        self.data = arr.view(-1, block_size)
        logger.info(f"总 tokens={len(all_ids)}, 块数={len(self.data)}, 每块长度={block_size}")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        block = self.data[idx]
        return block[:-1], block[1:]

# -----------------------------
# 模型：简化 GPT-like（带固定正弦/余弦位置编码）
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        # 上三角 mask 确保因果注意力（未来 token 不影响当前）
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
        x = x + self.attn(self.ln1(x))  # 残差连接（注意力）
        x = x + self.mlp(self.ln2(x))   # 残差连接（前馈）
        return x

class GPTLike(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, d_model, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # 共享输入输出嵌入权重
        self.block_size = block_size

        # 固定正弦/余弦位置编码（注册为 buffer，免梯度）
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
    parser = argparse.ArgumentParser(description="简化 GPT-like 模型训练（DeepSpeed Stage-2，WikiText-2）")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="本地 DataLoader 批次大小")
    parser.add_argument("--block_size", type=int, default=512, help="序列块大小（<=512）")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n_layer", type=int, default=12, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=16, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=1024, help="模型维度（必须能被 n_head 整除）")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")
    parser.add_argument("--ds_config", type=str, default="ds_config.json", help="DeepSpeed 配置文件路径")
    parser.add_argument("--local_rank", type=int, default=0, help="由 DeepSpeed launcher 自动传入")
    args = parser.parse_args()

    # 参数校验
    if args.block_size > 512:
        raise ValueError("block_size 超过最大 512")
    if args.d_model % args.n_head != 0:
        raise ValueError("d_model 必须能被 n_head 整除（保证 head dim 为整数）")

    # 读取 DeepSpeed 配置文件的 batch size（如存在）
    ds_batch = None
    if args.ds_config and os.path.isfile(args.ds_config):
        with open(args.ds_config, "r") as f:
            ds_conf = json.load(f)
        ds_batch = ds_conf.get("train_micro_batch_size_per_gpu")
        if ds_batch is not None:
            ds_batch = int(ds_batch)

    # DeepSpeed 依赖检查
    if deepspeed is None:
        logger.error("未检测到 deepspeed，请先安装：pip install deepspeed")
        return

    # 初始化 DeepSpeed 分布式环境
    logger.info("开始初始化分布式训练...")
    world_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = (world_rank == 0)
    try:
        deepspeed.init_distributed()
        if is_main:
            logger.info("DeepSpeed 分布式初始化成功")
    except Exception as e:
        logger.error(f"DeepSpeed 分布式初始化失败: {e}")
        raise
    if is_main:
        logger.info(f"DeepSpeed 环境：rank={world_rank}, world_size={world_size}")

    # 设置设备
    logger.info("初始化设备...")
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if is_main:
        logger.info(f"使用设备: {device}")

    # 使用 DeepSpeed 配置的 batch size 或命令行 batch_size
    chosen_batch = ds_batch if ds_batch is not None else args.batch_size
    logger.info(f"最终 DataLoader 本地 batch_size={chosen_batch}（每进程）")

    # 加载数据与分词器
    logger.info("开始加载数据集...")
    texts = prepare_data()
    logger.info("数据集加载完成，开始初始化分词器...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # 添加 pad token（若无）
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    vocab_size = len(tokenizer)
    if is_main:
        logger.info(f"分词器初始化完成，词表大小: {vocab_size}")

    logger.info("开始初始化数据集...")
    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size)
    logger.info("数据集初始化完成，开始创建 DataLoader...")
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=chosen_batch, sampler=sampler,
                            shuffle=(sampler is None), drop_last=True, num_workers=4)
    if is_main:
        logger.info("DataLoader 初始化完成")

    # 初始化模型与优化器
    logger.info("开始初始化模型...")
    model = GPTLike(vocab_size, args.block_size, args.n_layer, args.n_head, args.d_model, args.dropout)
    model = model.to(device)
    logger.info("模型移动到设备完成")

    # 调整 token embedding 和 head 权重以匹配 vocab_size
    model.tok_emb = nn.Embedding(vocab_size, args.d_model).to(device)
    model.head = nn.Linear(args.d_model, vocab_size, bias=False).to(device)
    model.head.weight = model.tok_emb.weight
    logger.info("模型嵌入层调整完成")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    logger.info("开始 DeepSpeed 初始化...")
    engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        config=args.ds_config
    )
    if is_main:
        logger.info("DeepSpeed 引擎初始化完成")

    Path("checkpoints").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # 训练循环
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        engine.train()
        total_loss, num_batches = 0.0, 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            logits = engine(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            engine.backward(loss)
            engine.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0 and is_main:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}, Loss={loss.item():.4f}")

        if num_batches > 0 and is_main:
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Avg Loss={total_loss/num_batches:.4f}")

        # 保存 checkpoint
        engine.save_checkpoint("checkpoints", tag=f"epoch{epoch+1}")
        if is_main:
            logger.info(f"已保存检查点: checkpoints/epoch{epoch+1}")

    # 保存最终模型
    if is_main:
        torch.save(engine.module.state_dict(), "models/final_model.pth")
        logger.info("最终模型已保存至 models/final_model.pth")

    # 优雅关闭分布式环境
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            if is_main:
                logger.info("已销毁 torch.distributed 进程组")
    except Exception as e:
        logger.warning(f"销毁分布式进程组时出错: {e}")


if __name__ == "__main__":
    main()
