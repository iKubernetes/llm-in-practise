#!/usr/bin/env python3
"""
简化的 GPT-like 模型训练脚本（FSDP2 - PyTorch 2.8+ 教学版）
使用 fully_shard, DeviceMesh, MixedPrecisionPolicy 和 OffloadPolicy。

使用示例（多GPU/多机建议用 torchrun）:
    # 单机 2 GPU
    torchrun --nproc_per_node=2 fsdp2_gpt_wikitext2.py --epochs 3 --batch_size 8

    # 两台机器，每台 2 GPU（假设已经设置 MASTER_ADDR/MASTER_PORT 等）
    # export MASTER_ADDR="aihost1.magedu.com"
    # export MASTER_PORT=29500
    # 在第一台机器 (node_rank=0)
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 fsdp2_gpt_wikitext2.py --epochs 3 --batch_size 8
    # 在第二台机器 (node_rank=1)
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 fsdp2_gpt_wikitext2.py --epochs 3 --batch_size 8

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

# FSDP2 核心导入 (适用于 PyTorch 2.8+)
from torch.distributed.fsdp import (
    fully_shard,              
    ShardingStrategy,
    OffloadPolicy,            
    MixedPrecisionPolicy,     
)
# 修正 'transformer_auto_wrap_policy' 的导入路径
# 注意: auto_wrap_policy 在 fully_shard 中已不再作为参数
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# 导入 DeviceMesh 和 Checkpoint 相关模块
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions

from transformers import BertTokenizer
from datasets import load_dataset

# 配置日志 (保持不变)
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =================================================================
# 数据集与模型定义 (完整)
# =================================================================
def prepare_data():
    """加载 WikiText 数据集"""
    logger.info("加载 WikiText 数据集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex["text"] for ex in dataset if ex.get("text") and ex["text"].strip()]
    logger.info(f"加载 {len(texts)} 条非空文本")
    return texts

class TokenizedDataset(Dataset):
    """处理后的 Tokenized Dataset"""
    def __init__(self, texts, tokenizer, block_size=256):
        self.block_size = block_size
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

class CausalSelfAttention(nn.Module):
    """Causal Self-Attention 模块"""
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
    """Feed-Forward 网络模块"""
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
    """Transformer Block (GPT 风格)"""
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
    """简化的 GPT-Like 模型"""
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

# =================================================================
# 分布式工具函数 (保持不变)
# =================================================================
def setup_distributed(args):
    """根据环境与参数初始化分布式"""
    local_rank = int(os.environ.get("LOCAL_RANK", -1)) if args.local_rank is None else args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
    return is_distributed, local_rank, rank, world_size

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

# =================================================================
# 主训练流程 (FSDP2/fully_shard 配置 - 最终版)
# =================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="简化的 GPT-like 模型训练（FSDP2 教学版）")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="每个进程的批次大小")
    parser.add_argument("--block_size", type=int, default=256, help="序列块大小（不得超过 512）")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=12, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=768, help="模型维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")
    parser.add_argument("--local_rank", type=int, default=None, help="本地 GPU id（由启动器自动传入）")
    args = parser.parse_args()

    if args.block_size > 512:
        raise ValueError(f"block_size ({args.block_size}) 超过 BERT 最大序列长度 512")
    if args.d_model % args.n_head != 0:
        raise ValueError(f"d_model ({args.d_model}) 必须能被 n_head ({args.n_head}) 整除")

    is_distributed, local_rank, rank, world_size = setup_distributed(args)

    if torch.cuda.is_available():
        if is_distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank if is_distributed else 0}")
    else:
        device = torch.device("cpu")

    if not is_distributed or rank == 0:
        logger.info(f"使用设备: {device}")

    texts = prepare_data() 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

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
        # ------------ FSDP2/fully_shard 配置开始 ------------

        # 1. 定义设备网格 (DeviceMesh)
        mesh = DeviceMesh(device_type="cuda", mesh=[rank for rank in range(world_size)])

        # 2. 混合精度策略 (MixedPrecisionPolicy)
        # 此对象仅用于定义 dtype，不再传递给 fully_shard
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        
        mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch_dtype,        
            reduce_dtype=torch_dtype,       
            cast_forward_inputs=True,       
        )

        # 3. CPU Offload 策略 (OffloadPolicy)
        #cpu_offload_policy = None 
        cpu_offload_policy = OffloadPolicy() 

        # 4. 自动包裹策略（定义但不再传入 fully_shard）
        # 如果需要分层 FSDP，需要在 fully_shard 之前显式使用 ShardingSpec 或其他方法
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )
        
        # 5. FSDP2 (fully_shard) 封装模型
        logger.info("使用 FSDP2 (fully_shard) 封装模型...")
        model_to_train = fully_shard(
            module=model,
            mesh=mesh,                                      
            offload_policy=cpu_offload_policy,              
            # 移除所有在最新 API 中被废弃的参数:
            # mixed_precision=..., 
            # auto_wrap_policy=..., 
        )
        logger.info("FSDP2 模型封装完成。")
        # ------------ FSDP2/fully_shard 配置结束 ------------
    else:
        model_to_train = model

    optimizer = optim.AdamW(model_to_train.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if not is_distributed or rank == 0:
        Path("checkpoints").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

    # 训练循环 
    for epoch in range(args.epochs):
        model_to_train.train()
        if is_distributed:
            sampler.set_epoch(epoch)

        total_loss, num_batches = 0.0, 0
        batch_losses = []

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model_to_train(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            batch_losses.append(loss_val)
            total_loss += loss_val
            num_batches += 1

            if ((batch_idx + 1) % 50 == 0) and (not is_distributed or rank == 0):
                avg_loss = sum(batch_losses[-50:]) / min(len(batch_losses), 50)
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Avg Loss (last 50): {avg_loss:.4f}")

        avg_loss = total_loss / max(1, num_batches)
        if not is_distributed or rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")

        # 检查点保存 (兼容 FSDP2)
        if is_distributed:
            model_state_dict, _ = get_state_dict(
                model_to_train,
                optimizers=(),  
                options=StateDictOptions(full_state_dict=True)
            )
            if rank == 0:
                torch.save(model_state_dict, f"checkpoints/model_epoch_{epoch+1}.pth")
        else:
            torch.save(model_to_train.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

    # 最终模型保存
    if is_distributed:
        model_state_dict, _ = get_state_dict(
            model_to_train,
            optimizers=(), 
            options=StateDictOptions(full_state_dict=True)
        )
        if rank == 0:
            torch.save(model_state_dict, "models/final_model.pth")
            logger.info("最终模型已保存至 models/final_model.pth")
    else:
        torch.save(model_to_train.state_dict(), "models/final_model.pth")
        logger.info("最终模型已保存至 models/final_model.pth")

    if is_distributed:
        cleanup_distributed()

if __name__ == "__main__":
    main()
