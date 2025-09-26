#!/usr/bin/env python3
"""
Usage:
  torchrun --nproc_per_node=2 ddp_gpt_bpe_tokenizer_02.py --epochs 3 --per_device_batch_size 8 --block_size 128
"""
import os
import time
import math
import argparse
import random
import logging
from typing import List, Iterator, Optional, Dict, Any
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset

from transformers import PreTrainedTokenizerBase, AutoTokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import trainers, pre_tokenizers, processors, decoders, normalizers

# -----------------------------
# 配置与参数解析
# -----------------------------
def setup_logging(rank: int, save_dir: str = None):
    """为每个rank设置日志，主进程同时输出到文件和终端"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - RANK %(d} - %(levelname)s - %(message)s'.format(d=rank))

    # 所有rank都输出到控制台（但torchrun可能只显示rank0）
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 主进程额外输出到文件
    if rank == 0 and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "training.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def parse_args():
    p = argparse.ArgumentParser(description="优化版 DDP GPT 训练脚本")
    p.add_argument("--epochs", type=int, default=3, help="训练轮数")
    p.add_argument("--per_device_batch_size", type=int, default=8, help="每个GPU的批次大小")
    p.add_argument("--block_size", type=int, default=128, help="输入序列长度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--save_dir", type=str, default="./checkpoints_ddp_optimized", help="模型和日志保存目录")
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="用于torchrun自动注入")

    # Tokenizer 选项
    p.add_argument("--train_tokenizer", action="store_true", help="在Wikitext上训练一个新的BPE分词器")
    p.add_argument("--tokenizer_path", type=str, default="./optimized_tokenizer.json", help="分词器保存/加载路径")
    p.add_argument("--vocab_size", type=int, default=30000, help="训练BPE时的词汇表大小")
    p.add_argument("--use_pretrained_tokenizer", type=str, default="", help="使用Transformers的预训练分词器名称 (e.g. 'gpt2')")

    # 训练优化选项
    p.add_argument("--val_split_ratio", type=float, default=0.05, help="验证集划分比例")
    p.add_argument("--eval_interval", type=int, default=500, help="每隔多少步评估一次验证集")
    p.add_argument("--early_stop_patience", type=int, default=None, help="早停耐心值 (None表示禁用)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数，用于模拟更大batch size")
    
    return p.parse_args()

# -----------------------------
# 分布式训练助手函数
# -----------------------------
def setup_distributed():
    """初始化分布式训练环境，自动识别rank和world_size"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1
        logging.info("未发现分布式环境变量，以单进程模式运行")

    # 只有多个进程时才需要初始化进程组
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(seconds=30))
            logging.info(f"分布式初始化完成: rank={rank}, world_size={world_size}, backend={backend}")
        except Exception as e:
            logging.error(f"分布式初始化失败: {e}")
            world_size = 1
            rank = 0

    return rank, world_size

def cleanup_distributed():
    """安全清理分布式进程组"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            logging.info("分布式进程组已清理")
    except Exception as e:
        logging.warning(f"清理分布式进程组时出错: {e}")

# -----------------------------
# 分词器处理
# -----------------------------
def setup_tokenizer(args, rank: int, world_size: int) -> PreTrainedTokenizerBase:
    """
    为所有rank设置分词器。确保仅由主进程执行文件操作，其他进程通过barrier等待并加载。
    返回一个兼容 transformers.PreTrainedTokenizerBase 接口的对象。
    """
    tokenizer = None
    tokenizer_name_or_path = ""

    # 确保所有进程在进入分词器设置前同步，避免竞争
    if world_size > 1:
        dist.barrier()

    try:
        if args.use_pretrained_tokenizer:
            # 使用 Transformers 预训练分词器
            tokenizer_name_or_path = args.use_pretrained_tokenizer
            if rank == 0:
                logging.info(f"正在加载预训练分词器: {tokenizer_name_or_path}")
            # 所有rank都加载
            hf_tokenizer = AutoTokenizer.from_pretrained(args.use_pretrained_tokenizer, use_fast=True)
            if hf_tokenizer.pad_token is None:
                hf_tokenizer.pad_token = hf_tokenizer.eos_token or "<|pad|>"
                logging.info(f"已设置 pad_token 为: {hf_tokenizer.pad_token}")
            tokenizer = hf_tokenizer
            vocab_size_msg = f", vocab_size={len(hf_tokenizer)}"

        else:
            # 使用 tokenizers 库训练或加载 BPE 分词器
            tokenizer_name_or_path = args.tokenizer_path
            if args.train_tokenizer:
                # 训练新模式：仅在rank 0上进行训练和保存
                if rank == 0:
                    if not os.path.exists(args.tokenizer_path):
                        logging.info("开始训练新的 BPE 分词器...")
                        texts_iter = wikitext_texts_iterator()
                        tokenizer = train_bpe_tokenizer_on_texts(texts_iter, args.vocab_size, args.tokenizer_path)
                        logging.info(f"分词器训练完成并已保存至 {args.tokenizer_path}")
                    else:
                        logging.info(f"分词器文件已存在，跳过训练: {args.tokenizer_path}")
                        tokenizer = load_tokenizer_from_path(args.tokenizer_path)
                # 其他rank等待主进程完成文件操作
                if world_size > 1:
                    dist.barrier()
                # 所有rank（包括刚训练完的主进程）都需要加载分词器
                if rank != 0: # 主进程在上面已经加载了
                    tokenizer = load_tokenizer_from_path(args.tokenizer_path)
            else:
                # 直接加载现有分词器
                if rank == 0:
                    logging.info(f"正在加载分词器: {args.tokenizer_path}")
                tokenizer = load_tokenizer_from_path(args.tokenizer_path)
            vocab_size_msg = f", vocab_size={tokenizer.get_vocab_size()}"

        if rank == 0:
            logging.info(f"分词器设置完成: {tokenizer_name_or_path}{vocab_size_msg}")

    except Exception as e:
        logging.error(f"设置分词器时发生错误: {e}")
        raise e

    # 再次同步，确保所有进程都成功加载分词器
    if world_size > 1:
        dist.barrier()

    return tokenizer

def train_bpe_tokenizer_on_texts(texts_iter: Iterator[str], vocab_size: int, save_path: str) -> HFTokenizer:
    """训练 BPE 分词器并保存"""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    # 使用底层组件创建，提供更多灵活性
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()]) # 可选的规范化
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True) # 类似GPT
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True
    )
    
    tokenizer.train_from_iterator(texts_iter, trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer

def load_tokenizer_from_path(path: str) -> HFTokenizer:
    """从文件加载 tokenizers 库的分词器"""
    return HFTokenizer.from_file(path)

def wikitext_texts_iterator(split: str = "train"):
    """从 HuggingFace datasets 加载 Wikitext 文本迭代器"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    for example in dataset:
        text = example["text"]
        if text.strip(): # 过滤空文本
            yield text

# -----------------------------
# 数据集与模型定义
# -----------------------------
class TokenizedBlocksDataset(Dataset):
    """使用分词器将文本编码并分块为固定长度的序列"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, texts: List[str], block_size: int):
        self.block_size = block_size
        token_ids = []

        # 统一编码接口
        if hasattr(tokenizer, 'encode'): 
            # 假设是 tokenizers.Tokenizer
            for text in texts:
                if text.strip():
                    encoded = tokenizer.encode(text)
                    token_ids.extend(encoded.ids)
        else:
            # 假设是 transformers.PreTrainedTokenizer
            for text in texts:
                if text.strip():
                    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
                    token_ids.extend(encoded["input_ids"])

        total_len = (len(token_ids) // block_size) * block_size
        self.blocks = []
        for i in range(0, total_len, block_size):
            self.blocks.append(torch.tensor(token_ids[i:i+block_size], dtype=torch.long))
        
        # 统一获取词汇表大小
        if hasattr(tokenizer, 'get_vocab_size'):
            self.vocab_size = tokenizer.get_vocab_size()
        else:
            self.vocab_size = len(tokenizer)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]

# ... (GPTLike, CausalSelfAttention, FeedForward, TransformerBlock 等模型定义与之前相同)
# 确保在 GPTLike 类中包含 resize_token_embeddings 方法

def prepare_datasets_and_loaders(args, tokenizer: PreTrainedTokenizerBase, rank: int, world_size: int):
    """准备训练集和验证集的数据加载器"""
    # 加载原始数据集
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex["text"] for ex in dataset if ex["text"].strip()]

    # 创建整体数据集并划分训练/验证集
    full_dataset = TokenizedBlocksDataset(tokenizer, texts, args.block_size)
    dataset_size = len(full_dataset)
    val_size = int(args.val_split_ratio * dataset_size)
    train_size = dataset_size - val_size

    # 使用随机种子确保所有进程划分一致
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # 创建分布式采样器 (训练集需要shuffle，验证集不需要)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_batch_size,
        sampler=val_sampler,
        drop_last=False, # 验证集不需要drop last
        pin_memory=torch.cuda.is_available(),
        num_workers=2
    )

    return train_loader, val_loader, train_sampler, val_sampler, full_dataset.vocab_size

# -----------------------------
# 训练与验证循环
# -----------------------------
@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device):
    """在给定数据加载器上评估模型，返回平均损失"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        inputs = batch.to(device, non_blocking=True)
        targets = inputs.clone()

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(inputs)
            B, L, V = logits.shape
            loss = F.cross_entropy(logits.view(B*L, V), targets.view(B*L), reduction="sum")

        total_loss += loss.item()
        total_samples += B

    # 如果是在分布式环境下，需要汇总所有进程的损失和样本数
    if dist.is_initialized():
        # 将所有进程的 total_loss 和 total_samples 汇总到设备0
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)

        dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_samples_tensor, dst=0, op=dist.ReduceOp.SUM)

        if dist.get_rank() == 0:
            avg_loss = total_loss_tensor.item() / total_samples_tensor.item()
        else:
            avg_loss = 0.0 # 其他进程不需要这个值
        # 将平均损失广播回所有进程，以便它们都能记录或用于早停判断（如果需要）
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.broadcast(avg_loss_tensor, src=0)
        avg_loss = avg_loss_tensor.item()
    else:
        avg_loss = total_loss / total_samples

    model.train()
    return avg_loss

def save_checkpoint(state: Dict[str, Any], save_dir: str, filename: str, is_main: bool, logger: logging.Logger):
    """保存检查点，仅在主进程执行"""
    if not is_main:
        return
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    logger.info(f"检查点已保存: {filepath}")

def load_checkpoint(ckpt_path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, scaler, device: torch.device, logger: logging.Logger):
    """加载检查点以恢复训练"""
    if not os.path.exists(ckpt_path):
        logger.warning(f"检查点文件不存在: {ckpt_path}")
        return None, 0, 0, float('inf')
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    if scaler and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    global_step = checkpoint.get('global_step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # 加载随机状态以确保可重现性（仅在单GPU或分布式rank0上有效，分布式环境下需要更复杂的处理）
    if 'random_state' in checkpoint:
        random.setstate(checkpoint['random_state'])
    if 'torch_random_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_random_state'].to('cpu'))
    if 'cuda_random_state' in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])

    logger.info(f"已从检查点恢复: {ckpt_path}, 将从epoch {start_epoch}开始")
    return start_epoch, global_step, best_val_loss

def train_epoch(model, train_loader, optimizer, scheduler, scaler, gradient_accumulation_steps, device, global_step, logger):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch.to(device, non_blocking=True)
        targets = inputs.clone()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(inputs)
            B, L, V = logits.shape
            loss = F.cross_entropy(logits.view(B*L, V), targets.view(B*L), reduction="mean")
            loss = loss / gradient_accumulation_steps # 梯度累积

        # 使用混合精度进行反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        epoch_loss += loss.item() * gradient_accumulation_steps # 累积前是平均过的

        # 梯度累积：达到累积步数时更新权重
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            global_step += 1

    avg_epoch_loss = epoch_loss / len(train_loader)
    return avg_epoch_loss, global_step

# -----------------------------
# 主函数
# -----------------------------
def main():
    # 解析参数和初始化分布式
    args = parse_args()
    rank, world_size = setup_distributed()
    is_main = (rank == 0)

    # 设置设备
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 设置日志
    logger = setup_logging(rank, args.save_dir)
    if is_main:
        logger.info(f"开始优化版DDP GPT训练，参数: {args}")

    # 设置随机种子（注意每个rank的种子不同）
    seed = args.seed + rank
    set_seed(seed)
    logger.info(f"Rank {rank} 随机种子设置为: {seed}")

    # 设置分词器
    try:
        tokenizer = setup_tokenizer(args, rank, world_size)
        # 统一获取词汇表大小
        if hasattr(tokenizer, 'get_vocab_size'):
            vocab_size_from_tokenizer = tokenizer.get_vocab_size()
        else:
            vocab_size_from_tokenizer = len(tokenizer)
    except Exception as e:
        logger.error(f"初始化分词器失败: {e}")
        cleanup_distributed()
        return

    # 准备数据
    train_loader, val_loader, train_sampler, val_sampler, vocab_size_from_dataset = prepare_datasets_and_loaders(args, tokenizer, rank, world_size)
    # 以分词器的词汇表大小为准，因为模型需要与之匹配
    final_vocab_size = vocab_size_from_tokenizer
    if is_main:
        logger.info(f"最终确定词汇表大小: {final_vocab_size} (来自分词器)")

    # 创建模型
    model = GPTLike(vocab_size=final_vocab_size, block_size=args.block_size)
    # 如果词汇表大小与初始设置不同（例如使用了预训练分词器），调整模型
    if model.vocab_size != final_vocab_size:
        if is_main:
            logger.info(f"调整模型词汇表大小从 {model.vocab_size} 到 {final_vocab_size}")
        model.resize_token_embeddings(final_vocab_size)
    model.to(device)

    # 使用DDP包装模型
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # 优化器、混合精度、学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_training_steps = steps_per_epoch * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_training_steps)

    # 恢复训练（如果找到最新的检查点）
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    latest_ckpt_path = os.path.join(args.save_dir, "latest_checkpoint.pt")

    if os.path.exists(latest_ckpt_path):
        start_epoch, global_step, best_val_loss = load_checkpoint(latest_ckpt_path, model, optimizer, scheduler, scaler, device, logger)

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch) # 确保每个epoch数据顺序不同

        # 训练一个epoch
        train_loss, global_step = train_epoch(model, train_loader, optimizer, scheduler, scaler, args.gradient_accumulation_steps, device, global_step, logger)
        epoch_time = time.time() - epoch_start_time

        # 评估验证集
        if epoch % 1 == 0: # 每个epoch都验证，可根据需要调整
            val_loss = evaluate_model(model, val_loader, device)
            if is_main and val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # 保存最佳模型
                best_state = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                save_checkpoint(best_state, args.save_dir, "best_model.pt", is_main, logger)
            else:
                if is_main:
                    epochs_no_improve += 1

        # 保存最新检查点 (包含优化器、调度器等状态，便于恢复)
        if is_main:
            latest_state = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict() if scaler else None,
                'best_val_loss': best_val_loss,
                'random_state': random.getstate(),
                'torch_random_state': torch.get_rng_state(),
                'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            save_checkpoint(latest_state, args.save_dir, "latest_checkpoint.pt", is_main, logger)

            # 打印日志
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {best_val_loss:.4f}")

        # 早停判断
        if args.early_stop_patience is not None and epochs_no_improve >= args.early_stop_patience:
            if is_main:
                logger.info(f"早停触发，验证损失在 {epochs_no_improve} 个epoch内未提升")
            break

    if is_main:
        logger.info("训练完成！")
    cleanup_distributed()

if __name__ == "__main__":
    main()
