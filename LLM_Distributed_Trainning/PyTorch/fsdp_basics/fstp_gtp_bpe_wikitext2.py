#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDP GPT + BPE (WikiText-2) — 完整脚本（含 argparse、AMP 新 API、FSDP/保存修复、兼容 cleanup）
用法示例（推荐）:
  # 使用 torchrun (一进程一 GPU)
  torchrun --nproc_per_node=2 ddp_gpt_bpe_wikitext2_args.py --epochs 3 --batch_size 8

本脚本特性：
- argparse 支持常见超参数（epochs,batch_size,block_size,lr,vocab_size,...）
- 支持 torchrun（推荐）与 mp.spawn 两种启动方式
- 采用 HuggingFace tokenizers / ByteLevelBPETokenizer
- 修复 AMP 新 API 警告（torch.amp）
- 修复 cleanup 中 barrier(timeout) 在不同 PyTorch 版本上的兼容性
- 对 FSDP / state_dict 保存做了安全处理（所有 rank 都参与 collectives）
"""
import os
import sys
import time
import argparse
import logging
import traceback
import inspect
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# tokenizers / datasets
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset

# try import FSDP (optional)
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    FSDP_AVAILABLE = True
except Exception:
    FSDP = None
    FSDP_AVAILABLE = False

# disable tokenizers internal parallel (avoid fork warnings)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ddp_gpt_bpe")

# -----------------------------
# helper: get env ranks
# -----------------------------
def get_env_rank_local_world():
    """Read env vars set by torchrun. If not present, return (0,0,1)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size

# -----------------------------
# stable cleanup (compat with versions)
# -----------------------------
def cleanup_distributed():
    """Robust cleanup that handles different PyTorch versions and avoids barrier(timeout) TypeError."""
    try:
        if not dist.is_initialized():
            return
        # try to call barrier with timeout if signature supports it
        try:
            sig = inspect.signature(dist.barrier)
            if "timeout" in sig.parameters:
                # choose an appropriate timeout (timedelta)
                world_size = dist.get_world_size()
                timeout_sec = max(600, 60*10) * world_size
                dist.barrier(timeout=timedelta(seconds=timeout_sec))
            else:
                dist.barrier()
        except Exception as e:
            logger.warning("dist.barrier() with timeout unsupported or raised: %s", e)
            try:
                dist.barrier()
            except Exception:
                logger.warning("dist.barrier() fallback also failed; proceeding to destroy_process_group")

        try:
            dist.destroy_process_group()
            logger.info("Destroyed process group successfully")
        except Exception as e:
            logger.warning("dist.destroy_process_group() raised: %s", e)
    except Exception:
        logger.warning("cleanup_distributed outer exception:\n" + traceback.format_exc())

# -----------------------------
# data & tokenizer helpers
# -----------------------------
def prepare_wikitext_texts():
    logger.info("Loading WikiText-2 raw train split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex["text"] for ex in ds if isinstance(ex.get("text"), str) and ex["text"].strip()]
    logger.info("Loaded %d non-empty text entries", len(texts))
    return texts

def train_bpe_tokenizer_from_texts(texts_iter, vocab_size, save_dir):
    """Train ByteLevel BPE tokenizer and save files (vocab.json merges.txt) into save_dir."""
    logger.info("Rank 0: training BPE tokenizer ...")
    tok = ByteLevelBPETokenizer()
    # trainer params: you can tune min_frequency, special tokens etc.
    tok.train_from_iterator(texts_iter, vocab_size=vocab_size, min_frequency=2, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
    os.makedirs(save_dir, exist_ok=True)
    tok.save_model(save_dir)
    logger.info("Tokenizer saved to %s", save_dir)
    return tok

def load_or_train_tokenizer(tokenizer_dir, texts, vocab_size, rank, world_size):
    """
    If tokenizer exists -> load; else rank0 trains and saves, others barrier then load.
    Returns ByteLevelBPETokenizer instance.
    """
    if os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
        logger.info("[rank %d] Loading tokenizer from %s", rank, tokenizer_dir)
        tok = ByteLevelBPETokenizer.from_file(os.path.join(tokenizer_dir, "vocab.json"),
                                              os.path.join(tokenizer_dir, "merges.txt"))
        return tok

    # need to train
    if rank == 0:
        tok = train_bpe_tokenizer_from_texts(iter(texts), vocab_size=vocab_size, save_dir=tokenizer_dir)
    else:
        tok = None

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    # load on others
    if rank != 0:
        if not os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
            raise FileNotFoundError(f"Tokenizer files not found on rank {rank} after barrier")
        tok = ByteLevelBPETokenizer.from_file(os.path.join(tokenizer_dir, "vocab.json"),
                                              os.path.join(tokenizer_dir, "merges.txt"))
    logger.info("[rank %d] tokenizer ready (vocab_size=%d)", rank, tok.get_vocab_size())
    return tok

# -----------------------------
# dataset
# -----------------------------
class TokenizedBlocksDataset(Dataset):
    """Concatenate token ids and split into non-overlapping blocks; return x=block[:-1], y=block[1:]"""
    def __init__(self, texts, tokenizer, block_size):
        self.block_size = block_size
        self.blocks = []
        all_ids = []
        for t in texts:
            if not t or not isinstance(t, str):
                continue
            enc = tokenizer.encode(t)
            ids = enc.ids
            if ids:
                all_ids.extend(ids)
        total = (len(all_ids) // block_size) * block_size
        if total == 0:
            raise ValueError("No full blocks available after tokenization. Reduce block_size or add data.")
        for i in range(0, total, block_size):
            self.blocks.append(all_ids[i:i+block_size])
        logger.info("Prepared %d blocks", len(self.blocks))

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        block = self.blocks[idx]
        x = torch.tensor(block[:-1], dtype=torch.long)
        y = torch.tensor(block[1:], dtype=torch.long)
        return x, y

# -----------------------------
# model (small GPT-like)
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.dropout = nn.Dropout(resid_dropout)
    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        attn_mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model), nn.Dropout(dropout))
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

class GPTLike(nn.Module):
    def __init__(self, vocab_size=30000, block_size=128, n_layer=6, n_head=8, d_model=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size - 1, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head, mlp_ratio=4.0, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
    def forward(self, idx):
        B, L = idx.size()
        if L > (self.block_size - 1):
            idx = idx[:, : (self.block_size - 1)]
            L = idx.size(1)
        pos = torch.arange(L, device=idx.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -----------------------------
# safe checkpoint helpers (FSDP-aware)
# -----------------------------
def save_checkpoint_allranks(model, optimizer, path, rank, is_main):
    """所有 rank 都调用 model.state_dict()，仅主 rank 写磁盘。"""
    try:
        if dist.is_initialized():
            dist.barrier()
        sd = model.state_dict()
        opt_sd = optimizer.state_dict() if optimizer is not None else None
        if dist.is_initialized():
            dist.barrier()
        if is_main:
            torch.save({"model_state": sd, "optimizer_state": opt_sd}, path)
            logger.info("[rank %d] saved checkpoint to %s", rank, path)
        if dist.is_initialized():
            dist.barrier()
    except Exception:
        logger.error("save_checkpoint_allranks failed:\n%s", traceback.format_exc())
        raise

# -----------------------------
# train function (per-process)
# -----------------------------
def train_worker(args, rank, local_rank, world_size):
    """单个进程的训练入口（被 torchrun 或 mp.spawn 调用）"""
    logger.info(f"Starting worker rank={rank} local_rank={local_rank} world_size={world_size}")
    # set device early (NCCL requires)
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # init process group if needed
    if world_size > 1:
        try:
            dist.init_process_group(backend="nccl", init_method="env://")
            logger.info(f"[rank {rank}] init_process_group done")
        except Exception:
            logger.error("init_process_group failed:\n" + traceback.format_exc())
            raise

    is_main = (rank == 0)

    # seed
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    # load texts
    try:
        texts = prepare_wikitext_texts()
    except Exception:
        logger.error("Failed to load texts:\n" + traceback.format_exc())
        cleanup_distributed()
        return

    # tokenizer train/load
    try:
        tokenizer = load_or_train_tokenizer(args.tokenizer_dir, texts, args.vocab_size, rank, world_size)
        vocab_size = tokenizer.get_vocab_size()
    except Exception:
        logger.error("Tokenizer prepare failed:\n" + traceback.format_exc())
        cleanup_distributed()
        return

    # dataset + dataloader
    try:
        dataset = TokenizedBlocksDataset(texts, tokenizer, args.block_size)
    except Exception:
        logger.error("Failed to build dataset:\n" + traceback.format_exc())
        cleanup_distributed()
        return

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            shuffle=(sampler is None), num_workers=args.num_workers, pin_memory=True)

    # build model
    model = GPTLike(vocab_size=vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, d_model=args.d_model, dropout=args.dropout)
    model.to(device)

    # optionally wrap with FSDP if available & requested
    if args.use_fsdp and FSDP_AVAILABLE and world_size > 1:
        # simple whole-model wrap (safe). For large models, consider auto_wrap_policy.
        model = FSDP(model, device_id=local_rank)
        logger.info("[rank %d] model wrapped with FSDP", rank)
    else:
        if args.use_fsdp and not FSDP_AVAILABLE:
            logger.warning("FSDP requested but not available; proceeding without FSDP")

    # optimizer, criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # AMP: use new API (torch.amp) if available
    use_amp = args.use_amp and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # training loop
    try:
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            n_batches = 0
            start = time.time()
            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else _nullcontext():
                    logits = model(x)
                    B, L, V = logits.shape
                    loss = criterion(logits.view(B*L, V), y.view(B*L))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach().item())
                n_batches += 1
                if is_main and (batch_idx + 1) % args.log_interval == 0:
                    logger.info(f"Epoch {epoch+1} Batch {batch_idx+1} Loss {loss.item():.4f}")
            elapsed = time.time() - start
            avg_loss = total_loss / max(1, n_batches)
            if is_main:
                logger.info(f"Epoch {epoch+1} finished. avg_loss={avg_loss:.4f} elapsed={elapsed:.1f}s")
            # checkpoint: all ranks participate in state_dict to avoid FSDP collectives mismatch
            if (epoch + 1) % args.save_interval == 0:
                ckpt_name = f"data/ckpt_epoch{epoch+1}.pth"
                save_checkpoint_allranks(model, optimizer, ckpt_name, rank, is_main)

    except Exception:
        logger.error("Training loop failed:\n" + traceback.format_exc())
        raise
    finally:
        cleanup_distributed()

# -----------------------------
# small helper: nullcontext for non-amp case
# -----------------------------
from contextlib import contextmanager
@contextmanager
def _nullcontext():
    yield

# -----------------------------
# main / arg parsing / launch
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DDP GPT + BPE (WikiText-2) training")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8, help="per-process batch size")
    p.add_argument("--block_size", type=int, default=128, help="raw block length (inputs length = block_size-1)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vocab_size", type=int, default=30000)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tokenizer_dir", type=str, default="./bpe_tokenizer")
    p.add_argument("--save_interval", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision if available")
    p.add_argument("--use_fsdp", action="store_true", help="Wrap model with FSDP if available")
    p.add_argument("--spawn_procs", type=int, default=0, help="If >0 and not launched via torchrun, spawn this many procs locally")
    return p.parse_args()

def main():
    args = parse_args()
    # detect torchrun environment
    env_rank, env_local_rank, env_world = get_env_rank_local_world()
    if env_world > 1:
        # running under torchrun: each process is started separately
        rank = env_rank
        local_rank = env_local_rank
        world_size = env_world
        train_worker(args, rank, local_rank, world_size)
    else:
        # not under torchrun; if spawn_procs > 0, spawn processes locally with mp.spawn
        if args.spawn_procs > 0:
            world_size = args.spawn_procs
            def _spawn_fn(local_rank):
                rank = local_rank
                # set minimal env vars so child code can read them
                os.environ["RANK"] = str(rank)
                os.environ["LOCAL_RANK"] = str(local_rank)
                os.environ["WORLD_SIZE"] = str(world_size)
                train_worker(args, rank, local_rank, world_size)
            mp.spawn(_spawn_fn, nprocs=world_size, join=True)
        else:
            # single-process debug mode
            rank = 0
            local_rank = 0
            world_size = 1
            train_worker(args, rank, local_rank, world_size)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("Fatal error:\n" + traceback.format_exc())
        raise
