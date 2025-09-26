#!/usr/bin/env python3
"""
DDP GPT on Wikitext-2 (raw) — single-file example (no transformers)

使用说明（单机2GPU）:
  torchrun --nproc_per_node=2 ddp_gpt_wikitext2.py --epochs 3 --per_device_batch_size 8 --block_size 128

依赖项:
  pip install torch datasets

数据流程：
- 字节级分词：直接使用UTF-8字节作为token
- 文本拼接分块：将整个数据集拼接后分割为固定长度块
- 分布式数据加载：每个进程只加载分配到的数据部分

模型特点：
- 轻量级GPT架构（6层Transformer）
- 权重绑定：输出层与嵌入层共享权重
- 因果注意力：使用上三角掩码防止未来信息泄露
"""
import os
import time
import math
import argparse
import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器
from datasets import load_dataset  # Hugging Face数据集库

# ---------------------------
# 参数解析 & 分布式工具函数
# ---------------------------
def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3, help="训练轮数")
    p.add_argument("--per_device_batch_size", type=int, default=8, help="每个GPU的批次大小")
    p.add_argument("--block_size", type=int, default=128, help="输入序列长度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--save_dir", type=str, default="./checkpoints_ddp", help="模型保存目录")
    # LOCAL_RANK由torchrun自动设置，表示当前进程在节点内的序号
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    return p.parse_args()

def setup_distributed():
    """初始化分布式环境"""
    # torchrun会自动设置以下环境变量
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])  # 全局进程ID
        world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
    else:
        # 单机模式回退
        rank, world_size = 0, 1
    
    # 根据硬件选择后端：GPU用nccl，CPU用gloo
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size

def cleanup_distributed():
    """清理分布式资源"""
    try:
        dist.destroy_process_group()
    except Exception:
        pass

# ---------------------------
# 分词器（字节级别）
# ---------------------------
def text_to_bytes_ids(text: str) -> List[int]:
    """将文本转换为UTF-8字节ID列表"""
    # 将文本编码为UTF-8字节序列
    b = text.encode("utf-8", errors="ignore")
    # 转换为0-255的整数列表
    return list(b)

# ---------------------------
# 数据集：拼接文本并分块
# ---------------------------
class ByteBlocksDataset(Dataset):
    """字节级文本数据集，将文本拼接后分割为固定长度的块"""
    def __init__(self, texts: List[str], block_size: int):
        byte_list = []
        # 拼接所有文本并转换为字节ID
        for t in texts:
            ids = text_to_bytes_ids(t)
            byte_list.extend(ids)
            # 添加换行符作为分隔符（可选）
            byte_list.append(10)  # 10是换行符的ASCII码
        
        # 计算可整除的总长度
        total_len = (len(byte_list) // block_size) * block_size
        self.blocks = []
        # 分割为固定长度的块
        for i in range(0, total_len, block_size):
            chunk = byte_list[i:i+block_size]
            self.blocks.append(torch.tensor(chunk, dtype=torch.long))
        
        # 词汇表大小固定为256（字节级）
        self.vocab_size = 256

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        # 返回指定索引的文本块
        return self.blocks[idx]  # shape (block_size,)

# ---------------------------
# 小型GPT风格模型
# ---------------------------
class CausalSelfAttention(nn.Module):
    """因果自注意力机制（防止未来信息泄露）"""
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        # 多头注意力机制
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.dropout = nn.Dropout(resid_dropout)

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        # 创建因果掩码：上三角矩阵（不包括对角线）
        attn_mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
        # 应用带掩码的多头注意力
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return self.dropout(out)

class FeedForward(nn.Module):
    """前馈神经网络（两层线性变换+GELU激活）"""
    def __init__(self, d_model, hidden_dim, dropout=0.0):
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
    """Transformer块（自注意力+前馈网络）"""
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # 层归一化1
        self.attn = CausalSelfAttention(d_model, nhead, resid_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)  # 层归一化2
        hidden = int(d_model * mlp_ratio)  # 前馈网络隐藏层维度
        self.mlp = FeedForward(d_model, hidden, dropout=dropout)

    def forward(self, x):
        # 残差连接+自注意力
        x = x + self.attn(self.ln1(x))
        # 残差连接+前馈网络
        x = x + self.mlp(self.ln2(x))
        return x

class GPTLike(nn.Module):
    """GPT风格的语言模型"""
    def __init__(self, vocab_size=256, block_size=128, n_layer=6, n_head=8, d_model=512, dropout=0.1):
        super().__init__()
        # 词嵌入层
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # 位置嵌入层
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        # Transformer块堆叠
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, mlp_ratio=4.0, dropout=dropout) 
            for _ in range(n_layer)
        ])
        # 最终层归一化
        self.ln_f = nn.LayerNorm(d_model)
        # 输出层（与词嵌入层权重共享）
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # 权重绑定
        self.block_size = block_size

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化（GPT风格）"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx):
        """前向传播"""
        B, L = idx.size()  # 批次大小, 序列长度
        assert L <= self.block_size
        
        # 创建位置索引
        pos = torch.arange(L, device=idx.device).unsqueeze(0).expand(B, -1)
        # 词嵌入 + 位置嵌入
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        
        # 通过所有Transformer块
        for blk in self.blocks:
            x = blk(x)
        
        # 最终层归一化和输出
        x = self.ln_f(x)
        logits = self.head(x)
        return logits  # (B, L, V)

# ---------------------------
# 工具函数：检查点、随机种子、数据加载器初始化
# ---------------------------
def save_checkpoint(state, save_dir, step, is_main):
    """保存模型检查点（仅在主进程执行）"""
    if not is_main:
        return
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"ckpt_step{step}.pt")
    torch.save(state, fname)
    print(f"[rank 0] saved {fname}")

def set_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    """数据加载器工作进程初始化函数"""
    # 使用全局种子+worker_id确保不同worker有不同的随机序列
    seed = torch.initial_seed() % (2**32 - 1)
    random.seed(seed + worker_id)
    # 如果使用numpy，也需要设置种子
    try:
        import numpy as np
        np.random.seed(seed + worker_id)
    except Exception:
        pass

# ---------------------------
# 数据准备 & 数据加载器
# ---------------------------
def prepare_dataloader(block_size, per_device_batch_size, rank, world_size, num_workers=2):
    """准备分布式数据加载器"""
    # 加载Wikitext-2原始训练集
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x["text"] for x in ds]
    
    # 创建数据集
    dataset = ByteBlocksDataset(texts, block_size=block_size)
    
    # 创建分布式采样器（关键组件）
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,  # 总进程数
        rank=rank,                # 当前进程ID
        shuffle=True              # 启用随机打乱
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=per_device_batch_size, 
        sampler=sampler,          # 使用分布式采样器
        drop_last=True,            # 丢弃不完整的批次
        num_workers=num_workers,   # 数据加载工作进程数
        pin_memory=True,           # 锁页内存加速GPU传输
        worker_init_fn=worker_init_fn  # 工作进程初始化
    )
    return dataloader, dataset.vocab_size, sampler

# ---------------------------
# 训练循环
# ---------------------------
def train(args):
    """主训练函数"""
    # 初始化分布式环境
    rank, world_size = setup_distributed()
    # 获取本地rank（节点内的GPU序号）
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    
    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 判断是否是主进程（rank=0）
    is_main = (rank == 0)

    if is_main:
        print("Starting DDP GPT training")
        print("args:", args)

    # 设置随机种子（注意：每个rank使用不同的种子）
    set_seed(args.seed + rank)

    # 准备数据加载器
    dataloader, vocab_size, sampler = prepare_dataloader(
        args.block_size, 
        args.per_device_batch_size, 
        rank, 
        world_size
    )

    # 构建模型
    model = GPTLike(
        vocab_size=vocab_size, 
        block_size=args.block_size,
        n_layer=6, 
        n_head=8, 
        d_model=512, 
        dropout=0.1
    )
    model.to(device)

    # 使用DDP包装模型
    model = DDP(
        model, 
        device_ids=[local_rank] if torch.cuda.is_available() else None
    )

    # 优化器配置（区分权重衰减参数）
    no_decay = ["bias", "LayerNorm.weight"]
    params_with_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_no_decay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    
    optimizer = torch.optim.AdamW([
        {"params": params_with_decay, "weight_decay": args.weight_decay},
        {"params": params_no_decay, "weight_decay": 0.0}
    ], lr=args.lr)

    # 混合精度训练配置
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 计算总步数
    total_steps = math.ceil(len(dataloader) * args.epochs)
    if is_main:
        print(f"[rank 0] dataset size: {len(dataloader.dataset)}, steps_per_epoch: {len(dataloader)}, total_steps~{total_steps}")

    # 训练循环
    global_step = 0
    model.train()
    start = time.time()
    for epoch in range(args.epochs):
        # 关键：设置采样器的epoch（确保每个epoch数据顺序不同）
        sampler.set_epoch(epoch)
        
        if is_main:
            print(f"Epoch {epoch+1}/{args.epochs}")
        
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # 移动数据到当前设备（非阻塞传输）
            inputs = batch.to(device, non_blocking=True)
            targets = inputs.clone()  # 目标与输入相同（语言建模）

            # 混合精度训练上下文
            with torch.cuda.amp.autocast(enabled=use_amp):
                # 前向传播
                logits = model(inputs)  # (B, L, V)
                B, L, V = logits.shape
                # 计算交叉熵损失（展平序列）
                loss = F.cross_entropy(
                    logits.view(B*L, V), 
                    targets.view(B*L), 
                    reduction="mean"
                )

            # 反向传播（混合精度）
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # 记录损失
            loss_item = loss.detach().item()
            epoch_loss += loss_item
            global_step += 1

            # 主进程定期打印日志
            if is_main and global_step % 50 == 0:
                elapsed = time.time() - start
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"[step {global_step}] loss={loss_item:.4f} avg_epoch_loss={avg_loss:.4f} elapsed={elapsed:.1f}s")

            # 定期保存检查点
            if global_step % 1000 == 0:
                # 同步所有进程
                dist.barrier()
                if is_main:
                    ckpt = {
                        "step": global_step,
                        "epoch": epoch,
                        "model_state": model.module.state_dict(),  # 注意：使用.module访问原始模型
                        "optimizer_state": optimizer.state_dict(),
                    }
                    save_checkpoint(ckpt, args.save_dir, global_step, is_main)

        # 打印epoch统计信息
        if is_main:
            print(f"Epoch {epoch+1} finished. avg_loss={epoch_loss/len(dataloader):.4f}")

    # 最终检查点
    dist.barrier()
    if is_main:
        final_ckpt = {
            "step": global_step, 
            "epoch": args.epochs, 
            "model_state": model.module.state_dict(), 
            "optimizer_state": optimizer.state_dict()
        }
        save_checkpoint(final_ckpt, args.save_dir, global_step, is_main)
        print("Training completed.")
    
    # 清理分布式资源
    cleanup_distributed()

# ---------------------------
# 程序入口
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
