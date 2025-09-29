#!/usr/bin/env python3
"""
简化的 GPT-like 模型训练脚本（FSDP 教学版）
支持单机/单GPU、单机/多GPU、以及多机多GPU（基于 torch.distributed）.

使用示例（单机单GPU或CPU）:
    python fsdp_gtp_wikitext2.py --epochs 3 --batch_size 16

使用示例（多GPU/多机建议用 torchrun）:
    # 单机 2 GPU
    torchrun --nproc_per_node=2 fsdp_gtp_wikitext2.py --epochs 3 --batch_size 8

    # 两台机器，每台 2 GPU（假设已经设置 MASTER_ADDR/MASTER_PORT 等）
    # export MASTER_ADDR="aihost1.magedu.com"
    # export MASTER_PORT=29500
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 fsdp_gtp_wikitext2.py --epochs 3 --batch_size 8
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 fsdp_gtp_wikitext2.py --epochs 3 --batch_size 8

注意:
- --batch_size 是“每个进程”的 batch size（即每张卡的 batch size）
- 本脚本以教学与示例为主，追求简洁明了
"""

import os
import logging
import math
from pathlib import Path
import functools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer
from datasets import load_dataset

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

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
# 模型定义（与原脚本一致）
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
# 分布式工具函数
# -----------------------------
def setup_distributed(args):
    """根据环境与参数初始化分布式（如未检测到分布式环境，则返回 False）"""
    # local_rank 可以由 torchrun 注入为 --local_rank，或从环境变量 LOCAL_RANK 读取
    local_rank = int(os.environ.get("LOCAL_RANK", -1)) if args.local_rank is None else args.local_rank
    # world_size 和 rank 从环境变量读取（torchrun 会设置）
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    is_distributed = world_size > 1
    if is_distributed:
        # 使用 env:// 初始化（需设置 MASTER_ADDR/MASTER_PORT 或使用 torchrun）
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
    return is_distributed, local_rank, rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# -----------------------------
# 主训练流程
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="简化的 GPT-like 模型训练（FSDP 教学版）")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="每个进程的批次大小（per-process）")
    parser.add_argument("--block_size", type=int, default=256, help="序列块大小（不得超过 512）")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=12, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=768, help="模型维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")
    # 为兼容旧的 launcher，保留 local_rank 参数（torchrun 会注入）
    parser.add_argument("--local_rank", type=int, default=None, help="本地 GPU id（由 torch.distributed 启动器自动传入）")
    args = parser.parse_args()

    if args.block_size > 512:
        raise ValueError(f"block_size ({args.block_size}) 超过 BERT 最大序列长度 512")
    if args.d_model % args.n_head != 0:
        raise ValueError(f"d_model ({args.d_model}) 必须能被 n_head ({args.n_head}) 整除")

    # 初始化分布式（如果是多进程运行）
    is_distributed, local_rank, rank, world_size = setup_distributed(args)

    # 设备设置
    if torch.cuda.is_available():
        if is_distributed:
            # local_rank 由 torchrun 注入；若仍为 -1，则尝试从环境变量读取
            if local_rank is None or local_rank < 0:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 只有主进程打印启动信息
    if not is_distributed or rank == 0:
        logger.info(f"运行信息 is_distributed={is_distributed}, rank={rank}, world_size={world_size}, local_rank={local_rank}")
        logger.info(f"使用设备: {device}")

    # 加载数据和分词器（为了示例简洁，在每个进程都加载 tokenizer 与原始文本）
    texts = prepare_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    if not is_distributed or rank == 0:
        logger.info(f"BERT 分词器词汇大小: {vocab_size}")

    # 创建数据集与分布式采样器
    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size)

    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False  # sampler 控制 shuffle
    else:
        sampler = None
        shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # 初始化模型
    model = GPTLike(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout
    )

    model.to(device)

    if is_distributed:
        # FSDP 配置：提供分片策略的定义（使用默认值），并支持修改为其它策略（其它策略以注释格式同时提供）
        # 基于默认值提供自动包装策略的定义（支持修改为其它策略）
        # cpu_offload的值为False
        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})  # 默认自动包装策略：针对 TransformerBlock 进行包装
        # 其他自动包装策略示例（取消注释启用）：
        # from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        # auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1e5)  # 基于参数大小的自动包装

        sharding_strategy = ShardingStrategy.FULL_SHARD  # 默认分片策略：全参数分片
        # 其他分片策略（取消注释启用）：
        # sharding_strategy = ShardingStrategy.SHARD_GRAD_OP  # 只分片梯度操作
        # sharding_strategy = ShardingStrategy.NO_SHARD  # 无分片，等同于 DDP
        # sharding_strategy = ShardingStrategy.HYBRID_SHARD  # 混合分片（组内全分片，组间复制）

        mixed_precision = MixedPrecision(
            param_dtype=torch.float32,  # 参数精度
            reduce_dtype=torch.float32,  # 梯度通信精度
            buffer_dtype=torch.float32,  # 缓冲区精度
        )

        cpu_offload = CPUOffload(offload_params=False)  # CPU offload 为 False

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            cpu_offload=cpu_offload,
            device_id=local_rank if torch.cuda.is_available() else None,
        )
    model_to_train = model

    optimizer = optim.AdamW(model_to_train.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 仅主进程创建目录并保存
    if (not is_distributed) or rank == 0:
        Path("checkpoints").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

    # 训练循环
    for epoch in range(args.epochs):
        model_to_train.train()
        if is_distributed:
            # 每个 epoch 必须调用 sampler.set_epoch 以保证 shuffle 不同
            sampler.set_epoch(epoch)

        total_loss, num_batches = 0.0, 0
        batch_losses = []

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model_to_train(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            batch_losses.append(loss_val)
            total_loss += loss_val
            num_batches += 1

            # 仅主进程打印
            if ((batch_idx + 1) % 50 == 0) and ((not is_distributed) or rank == 0):
                avg_loss = sum(batch_losses[-50:]) / min(len(batch_losses), 50)
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}, Avg Loss (last 50): {avg_loss:.4f}")

        # epoch 结束，计算全局平均损失
        local_avg_loss = total_loss / max(1, num_batches)
        loss_tensor = torch.tensor(local_avg_loss, device=device)
        if is_distributed:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

        # 主进程打印并保存检查点
        if (not is_distributed) or rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
            # 如果使用 FSDP，保存 full state_dict
            if is_distributed:
                with FSDP.state_dict_type(model_to_train, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                    state_dict = model_to_train.state_dict()
            else:
                state_dict = model_to_train.state_dict()
            torch.save(state_dict, f"checkpoints/model_epoch_{epoch+1}.pth")

    # 最终模型保存（仅主进程）
    if (not is_distributed) or rank == 0:
        if is_distributed:
            with FSDP.state_dict_type(model_to_train, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                final_state = model_to_train.state_dict()
        else:
            final_state = model_to_train.state_dict()
        torch.save(final_state, "models/final_model.pth")
        logger.info("最终模型已保存至 models/final_model.pth")

    # 清理分布式环境
    if is_distributed:
        cleanup_distributed()

if __name__ == "__main__":
    main()
