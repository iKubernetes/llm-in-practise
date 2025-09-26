#!/usr/bin/env python3
"""
启动示例:
    torchrun --nproc_per_node=2 ddp_gpt_bpe_wikitext2_compat.py --epochs 3 --batch_size 16
"""

import os
import sys
import argparse
import logging
import math
import multiprocessing
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# 第三方库：用于分词和数据集加载
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

# 禁用 tokenizers 并行性以避免 fork 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志：支持动态日志级别
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# 分布式训练初始化与清理
# -----------------------------
def setup_distributed():
    """初始化分布式训练进程组（若 WORLD_SIZE > 1）。"""
    # 使用默认值确保环境变量未设置时不会崩溃
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        try:
            if not dist.is_available():
                raise RuntimeError("torch.distributed 不可用")
            if not dist.is_initialized():
                # 根据设备选择分布式后端：CUDA 用 nccl，否则用 gloo
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend)
            logger.info(f"[Rank {rank}] 分布式初始化完成 (world_size={world_size})")
        except Exception as e:
            logger.exception(f"[Rank {rank}] 分布式初始化失败: {str(e)}")
            raise
    else:
        logger.info("单进程（非分布式）模式")
    return rank, world_size

def cleanup_distributed():
    """清理分布式训练进程组。"""
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("分布式进程组已销毁")
    except Exception:
        logger.exception("分布式清理失败")

# -----------------------------
# AMP 兼容性辅助函数
# -----------------------------
def create_grad_scaler_for_device(device):
    """
    创建与 PyTorch 版本兼容的 GradScaler。
    优先使用 torch.amp.GradScaler('cuda')，回退到 torch.cuda.amp.GradScaler。
    若设备非 CUDA，则返回 None。
    """
    if device.type != "cuda":
        logger.info("设备为非 CUDA，无需 GradScaler")
        return None

    try:
        scaler = torch.amp.GradScaler("cuda")
        logger.info("使用 torch.amp.GradScaler")
        return scaler
    except Exception:
        pass

    try:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("使用 torch.cuda.amp.GradScaler")
        return scaler
    except Exception:
        logger.warning("未找到可用的 GradScaler 实现，混合精度训练可能被禁用")
        return None

@contextmanager
def autocast_for_device(device):
    """
    兼容不同 PyTorch 版本的自动混合精度上下文管理器。
    用法：
        with autocast_for_device(device):
            ...
    """
    if device.type != "cuda":
        yield
        return

    try:
        with torch.amp.autocast("cuda"):
            logger.debug("使用 torch.amp.autocast('cuda')")
            yield
            return
    except Exception:
        pass

    try:
        with torch.amp.autocast(device_type="cuda"):
            logger.debug("使用 torch.amp.autocast(device_type='cuda')")
            yield
            return
    except Exception:
        pass

    try:
        with torch.cuda.amp.autocast():
            logger.debug("使用 torch.cuda.amp.autocast")
            yield
            return
    except Exception:
        logger.warning("无法启用自动混合精度，降为无 AMP 模式")
        yield
        return

# -----------------------------
# 数据加载与分词
# -----------------------------
def prepare_data():
    """加载 wikitext-2-raw-v1 数据集的训练集，返回非空文本列表。"""
    logger.info("加载 WikiText 数据集 (wikitext-2-raw-v1)...")
    try:
        # 使用流式加载以减少内存占用
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
        texts = [ex["text"] for ex in dataset if ex.get("text") and ex["text"].strip()]
        logger.info(f"加载 {len(texts)} 条非空文本")
        return texts
    except Exception:
        logger.exception("数据集加载失败")
        raise

def train_bpe_tokenizer(texts, vocab_size=30000, save_path="bpe_tokenizer.json", batch_size=1000):
    """使用 tokenizers 库以流式/批量方式训练 BPE 分词器。"""
    logger.info(f"训练 BPE 分词器: vocab_size={vocab_size}, save_path={save_path}")
    try:
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
            show_progress=True
        )

        def iter_texts():
            for i in range(0, len(texts), batch_size):
                yield texts[i:i + batch_size]

        tokenizer.train_from_iterator(iter_texts(), trainer=trainer)
        tokenizer.save(save_path)
        logger.info(f"分词器已保存至 {save_path}")
        return tokenizer
    except Exception:
        logger.exception("分词器训练失败")
        raise

# -----------------------------
# 数据集：批量编码与块视图
# -----------------------------
class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer: Tokenizer, block_size=256, batch_limit=1024):
        """初始化数据集，批量编码文本并按块组织。"""
        super().__init__()
        self.block_size = block_size

        logger.info(f"批量编码文本 (batch_limit={batch_limit})...")
        all_ids = []
        batch = []
        for t in texts:
            batch.append(t)
            if len(batch) >= batch_limit:
                encs = tokenizer.encode_batch(batch)
                for e in encs:
                    all_ids.extend(e.ids)
                batch = []
        if batch:
            encs = tokenizer.encode_batch(batch)
            for e in encs:
                all_ids.extend(e.ids)

        total_len = (len(all_ids) // block_size) * block_size
        if total_len == 0:
            raise ValueError(f"分词后数据不足以构成 block_size={block_size} 的块")

        arr = torch.tensor(all_ids[:total_len], dtype=torch.long)
        self.data = arr.view(-1, block_size)
        logger.info(f"TokenizedDataset: 总 token 数={len(all_ids)}, 块数={len(self.data)}")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        block = self.data[idx]
        x = block[:-1].clone()  # 输入序列
        y = block[1:].clone()   # 目标序列（偏移一位）
        return x, y

# -----------------------------
# 模型定义 (GPT-like)
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, resid_dropout=0.0):
        """初始化因果自注意力模块。"""
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.dropout = nn.Dropout(resid_dropout)

    def forward(self, x):
        seq_len = x.size(1)
        # 创建因果掩码：上三角为 True（屏蔽未来 token）
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        """初始化前馈网络模块。"""
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
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1):
        """初始化 Transformer 块（包含自注意力和前馈网络）。"""
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, resid_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = FeedForward(d_model, hidden, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # 残差连接：注意力
        x = x + self.mlp(self.ln2(x))   # 残差连接：前馈
        return x

class GPTLike(nn.Module):
    def __init__(self, vocab_size=30000, block_size=256, n_layer=6, n_head=8, d_model=768, dropout=0.1):
        """初始化 GPT-like 模型。"""
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, mlp_ratio=4.0, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # 共享输入和输出嵌入权重
        self.head.weight = self.tok_emb.weight
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重：线性层和嵌入层使用正态分布，LayerNorm 初始化为标准值。"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 优化建议：可动态调整 std
        if isinstance(module, nn.Linear) and getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx):
        """前向传播：输入 token 索引，输出 logits。"""
        B, L = idx.size()
        if L > self.block_size:
            idx = idx[:, :self.block_size]
            L = self.block_size
        pos = torch.arange(L, device=idx.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -----------------------------
# 主训练流程
# -----------------------------
def main():
    """主函数：解析参数、设置环境、训练模型并保存检查点。"""
    parser = argparse.ArgumentParser(description="兼容版 BPE + DDP 训练")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--block_size", type=int, default=256, help="序列块大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--vocab_size", type=int, default=30000, help="分词器词汇大小")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=768, help="模型维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")
    parser.add_argument("--save_interval", type=int, default=1, help="检查点保存间隔")
    parser.add_argument("--num_gpus", type=int, default=None, help="预期 GPU 数量")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    args = parser.parse_args()

    # 参数验证
    if args.block_size <= 0:
        raise ValueError("block_size 必须大于 0")
    if args.vocab_size <= 0:
        raise ValueError("vocab_size 必须大于 0")

    # 设备信息
    total_cuda = torch.cuda.device_count()
    logger.info(f"检测到 CUDA 设备数: {total_cuda}")
    intended_world_size = args.num_gpus if args.num_gpus is not None else total_cuda
    intended_world_size = max(1, min(intended_world_size, max(1, total_cuda)))
    logger.info(f"预期 GPU 数（仅供参考）: {intended_world_size}")

    # 分布式初始化
    rank, world_size = setup_distributed()

    # 选择设备
    if torch.cuda.is_available() and world_size > 0:
        device_id = rank if rank < torch.cuda.device_count() else 0
        try:
            torch.cuda.set_device(device_id)
            logger.info(f"[Rank {rank}] 设置 CUDA 设备: cuda:{device_id}")
        except Exception:
            logger.warning(f"[Rank {rank}] 设置 cuda:{device_id} 失败，继续执行")
        device = torch.device(f"cuda:{device_id}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    logger.info(f"[Rank {rank}] 使用设备: {device}")

    # 设置随机种子
    seed = args.seed + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 创建保存目录
    save_dir = Path(args.save_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据和分词器
    texts = prepare_data()
    tokenizer_path = "bpe_tokenizer.json"
    if not Path(tokenizer_path).exists():
        tokenizer = train_bpe_tokenizer(
            texts,
            vocab_size=args.vocab_size,
            save_path=tokenizer_path,
            batch_size=1024  # 优化建议：可通过参数调整
        )
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        logger.info(f"从 {tokenizer_path} 加载分词器")

    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"分词器词汇大小: {vocab_size}")

    # 创建数据集和数据加载器
    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size, batch_limit=1024)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    num_workers = min(4, max(0, multiprocessing.cpu_count() - 1))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        drop_last=True
    )

    # 初始化模型
    model = GPTLike(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout
    ).to(device)

    if world_size > 1:
        ddp_kwargs = {"device_ids": [device.index]} if device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs, find_unused_parameters=True)  # 优化建议：添加 find_unused_parameters

    # 初始化优化器、损失函数、梯度缩放器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = create_grad_scaler_for_device(device)
    if scaler is None and device.type == "cuda":
        logger.info("未创建 GradScaler，混合精度训练可能被禁用")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # 训练循环
    try:
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # 前向传播（支持 AMP）
                with autocast_for_device(device):
                    logits = model(x)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))

                loss_val = float(loss.item())
                # 梯度累积
                loss = loss / args.accumulate_grad_batches

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 优化步骤（梯度累积完成后）
                if (batch_idx + 1) % args.accumulate_grad_batches == 0 or (batch_idx + 1) == len(dataloader):
                    # 梯度裁剪
                    if args.clip_grad_norm and args.clip_grad_norm > 0:
                        if scaler is not None:
                            try:
                                scaler.unscale_(optimizer)
                            except Exception:
                                pass
                        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                total_loss += loss_val
                num_batches += 1

                if rank == 0 and (batch_idx % 100 == 0):
                    logger.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss_val:.4f}")

            # epoch 总结与检查点保存（仅主进程）
            if rank == 0:
                avg_loss = total_loss / max(1, num_batches)
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr
                logger.info(f"Epoch {epoch} 完成. 平均损失={avg_loss:.4f}, 当前学习率={lr_now:.6g}")

                if (epoch % args.save_interval) == 0:
                    ckpt = {
                        "epoch": epoch,
                        "model_state": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "vocab_size": vocab_size,
                        "block_size": args.block_size,
                        "args": vars(args)
                    }
                    if scaler is not None:
                        try:
                            ckpt["scaler_state"] = scaler.state_dict()
                        except Exception:
                            logger.warning("无法获取 scaler.state_dict()")
                    ckpt_path = save_dir / f"model_epoch_{epoch}.pth"
                    torch.save(ckpt, ckpt_path)
                    logger.info(f"检查点已保存至 {ckpt_path}")

                    # 优化建议：清理旧检查点
                    if epoch > args.save_interval * 5:
                        old_ckpt = save_dir / f"model_epoch_{epoch - args.save_interval * 5}.pth"
                        if old_ckpt.exists():
                            old_ckpt.unlink()
                            logger.info(f"删除旧检查点: {old_ckpt}")

    except RuntimeError as e:
        logger.exception(f"训练失败: {str(e)}")
        raise
    except ValueError as e:
        logger.exception(f"参数错误: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"未知错误: {str(e)}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
