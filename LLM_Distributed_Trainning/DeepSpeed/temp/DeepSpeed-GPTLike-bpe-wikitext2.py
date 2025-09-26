#!/usr/bin/env python3
"""
使用 DeepSpeed 进行单机多卡分布式训练的 GPT-like 模型脚本
- 基于原始脚本（gpt_simple_train_fixed_pe.py）改编
- 支持 ZeRO-1、FP16 混合精度和分布式数据加载
- 使用 BPE 分词器和 wikitext-2-raw-v1 数据集
启动示例:
    deepspeed --num_gpus 2 DeepSpeed-GPTLike-bpe-wikitext2.py --epochs 3 --batch_size 32
"""
import os
import argparse
import logging
import math
import multiprocessing
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import deepspeed

from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data():
    """加载 wikitext-2-raw-v1 数据集的训练集，返回非空文本列表。"""
    logger.info("加载 WikiText 数据集 (wikitext-2-raw-v1)...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [ex["text"] for ex in dataset if ex.get("text") and ex["text"].strip()]
        logger.info(f"加载 {len(texts)} 条非空文本")
        return texts
    except Exception:
        logger.exception("数据集加载失败")
        raise

def train_bpe_tokenizer(texts, vocab_size=30000, save_path="bpe_tokenizer.json", batch_size=1000):
    """训练 BPE 分词器"""
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
            for txt in texts:
                yield txt

        tokenizer.train_from_iterator(iter_texts(), trainer=trainer)
        tokenizer.save(save_path)
        logger.info(f"分词器已保存至 {save_path}")
        return tokenizer
    except Exception:
        logger.exception("分词器训练失败")
        raise

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
        x = block[:-1].clone()
        y = block[1:].clone()
        return x, y

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.dropout = nn.Dropout(resid_dropout)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.dropout(out)

class FeedForward(nn.Module):
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
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, resid_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = FeedForward(d_model, hidden, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

def get_sinusoidal_embeddings(seq_len, d_model, device):
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class GPTLike(nn.Module):
    def __init__(self, vocab_size=30000, block_size=256, n_layer=6, n_head=8, d_model=768, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, mlp_ratio=4.0, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.block_size = block_size
        self.d_model = d_model

        pe = get_sinusoidal_embeddings(block_size, d_model, device=torch.device("cpu"))
        self.register_buffer("pos_emb", pe, persistent=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and getattr(module, "bias", None) is not None:
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
        x = tok_emb + pos_emb
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def main():
    parser = argparse.ArgumentParser(description="使用 DeepSpeed 进行 GPT-like 模型分布式训练")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="全局批次大小")
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
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="DeepSpeed 配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地 GPU 编号，由 DeepSpeed 自动设置")
    args = parser.parse_args()

    if args.block_size <= 0:
        raise ValueError("block_size 必须大于 0")
    if args.vocab_size <= 0:
        raise ValueError("vocab_size 必须大于 0")

    deepspeed.init_distributed(dist_backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank == 0:
        logger.info(f"分布式训练初始化: rank={rank}, world_size={world_size}")

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    save_dir = Path(args.save_dir)
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    texts = prepare_data()
    tokenizer_path = "bpe_tokenizer.json"
    if not Path(tokenizer_path).exists():
        if rank == 0:
            tokenizer = train_bpe_tokenizer(
                texts,
                vocab_size=args.vocab_size,
                save_path=tokenizer_path,
                batch_size=1024
            )
        torch.distributed.barrier()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if rank == 0:
        logger.info(f"从 {tokenizer_path} 加载分词器")

    vocab_size = tokenizer.get_vocab_size()
    if rank == 0:
        logger.info(f"分词器词汇大小: {vocab_size}")

    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size, batch_limit=1024)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // world_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=min(4, max(0, multiprocessing.cpu_count() - 1)),
        drop_last=True
    )

    model = GPTLike(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config="ds_config.json"  # 修改：直接指定配置文件路径
    )

    criterion = nn.CrossEntropyLoss()

    try:
        for epoch in range(args.epochs):
            model_engine.train()
            sampler.set_epoch(epoch)
            total_loss = 0.0
            num_batches = 0

            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(model_engine.device, non_blocking=True)
                y = y.to(model_engine.device, non_blocking=True)

                logits = model_engine(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))

                model_engine.backward(loss)
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    model_engine.clip_grad_norm(args.clip_grad_norm)
                model_engine.step()

                loss_val = float(loss.item())
                total_loss += loss_val
                num_batches += 1

                if batch_idx % 100 == 0 and rank == 0:
                    logger.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss_val:.4f}")

            avg_loss = total_loss / max(1, num_batches)
            lr_now = optimizer.param_groups[0]["lr"]
            if rank == 0:
                logger.info(f"Epoch {epoch} 完成. 平均损失={avg_loss:.4f}, 当前学习率={lr_now:.6g}")

            if epoch % args.save_interval == 0 and rank == 0:
                ckpt_path = save_dir / f"model_epoch_{epoch}"
                model_engine.save_checkpoint(ckpt_path, tag=f"epoch_{epoch}")
                logger.info(f"检查点已保存至 {ckpt_path}")

                if epoch > args.save_interval * 5:
                    old_ckpt = save_dir / f"model_epoch_{epoch - args.save_interval * 5}"
                    if old_ckpt.exists():
                        import shutil
                        shutil.rmtree(old_ckpt)
                        logger.info(f"删除旧检查点: {old_ckpt}")

            torch.distributed.barrier()

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
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()  # 新增：清理分布式进程组

if __name__ == "__main__":
    main()
