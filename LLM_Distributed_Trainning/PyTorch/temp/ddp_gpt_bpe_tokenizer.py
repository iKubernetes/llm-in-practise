#!/usr/bin/env python3
"""
DDP GPT (custom model) + HuggingFace Tokenizers (BPE) example.
使用PyTorch DDP进行分布式训练，并集成HuggingFace Tokenizers库处理BPE分词的GPT模型示例。

Usage examples (single machine, 2 GPUs):
  # 1) 使用预训练 GPT-2 tokenizer（需要 transformers）
  torchrun --nproc_per_node=2 ddp_gpt_bpe_tokenizer.py --use_pretrained_tokenizer gpt2 --epochs 3 --per_device_batch_size 8 --block_size 128

  # 2) 在 Wikitext 上训练一个 BPE tokenizer（只需 tokenizers）
  torchrun --nproc_per_node=2 ddp_gpt_bpe_tokenizer.py --train_tokenizer --vocab_size 30000 --epochs 3 --per_device_batch_size 8 --block_size 128

Requirements:
  pip install torch datasets tokenizers transformers
"""
import os
import time
import math
import argparse
import random
from typing import List, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler  # 分布式采样器
from datasets import load_dataset  # HuggingFace数据集库

# tokenizers 部分
from tokenizers import Tokenizer as HFTokenizer  # HuggingFace Tokenizers库
from tokenizers import models, trainers, pre_tokenizers, processors, normalizers, decoders
from transformers import AutoTokenizer  # Transformers库的自动分词器

# -----------------------------
# 参数解析和分布式辅助函数
# -----------------------------
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
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="本地进程排名")

    # tokenizer options - 分词器选项
    p.add_argument("--train_tokenizer", action="store_true", help="在Wikitext上训练一个新的BPE分词器")
    p.add_argument("--tokenizer_path", type=str, default="./tokenizer-bpe.json", help="分词器保存/加载路径")
    p.add_argument("--vocab_size", type=int, default=30000, help="训练BPE时的词汇表大小")
    p.add_argument("--use_pretrained_tokenizer", type=str, default="", help="通过transformers.AutoTokenizer加载的预训练分词器名称（如gpt2）")
    return p.parse_args()

def setup_distributed():
    """初始化分布式训练环境"""
    # 从环境变量获取rank和world_size（torchrun会自动设置）
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])  # 全局进程排名
        world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
    else:
        rank = 0
        world_size = 1  # 单机模式
    
    # 根据硬件选择后端：GPU用nccl，CPU用gloo
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size

def cleanup_distributed():
    """清理分布式训练资源"""
    try:
        dist.destroy_process_group()
    except Exception:
        pass

# -----------------------------
# 训练或加载分词器
# -----------------------------
def train_bpe_tokenizer_on_texts(texts_iter: Iterator[str], vocab_size: int, save_path: str):
    """
    在文本迭代器上训练BPE分词器（HF tokenizers）并保存到save_path（json格式）
    使用ByteLevel预分词器（类似RoBERTa/gpt2设置）
    
    参数:
        texts_iter: 文本迭代器
        vocab_size: 词汇表大小
        save_path: 保存路径
    """
    tokenizer = HFTokenizer(models.BPE(unk_token="[UNK]"))  # 创建BPE分词器
    # 设置规范化器（可选）
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()  # 字节级预分词器
    tokenizer.decoder = decoders.ByteLevel()  # 字节级解码器
    # 后处理器（如果需要添加bos/eos，这里我们不需要）
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
    tokenizer.train_from_iterator(texts_iter, trainer=trainer)  # 从迭代器训练
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.save(save_path)  # 保存分词器
    return tokenizer

def load_tokenizer_from_path(path: str):
    """从路径加载分词器"""
    return HFTokenizer.from_file(path)

# helper: iterate raw texts from wikitext train split (generator)
def wikitext_texts_iterator():
    """Wikitext文本迭代器（生成器）"""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    for ex in ds:
        txt = ex["text"]
        if isinstance(txt, str) and len(txt.strip())>0:
            yield txt

# -----------------------------
# 数据集：使用分词器编码并将文本分块
# -----------------------------
class TokenizedBlocksDataset(Dataset):
    """分词后的分块数据集"""
    def __init__(self, tokenizer, texts: List[str], block_size: int, add_special_tokens: bool = False):
        """
        初始化数据集
        
        参数:
            tokenizer: HFTokenizer (tokenizers库) 或 transformers.AutoTokenizer
            texts: 文本列表
            block_size: 目标序列长度（token数）
            add_special_tokens: 是否添加特殊token
        """
        self.block_size = block_size
        token_ids = []
        
        # 分批编码以避免内存爆炸
        for txt in texts:
            if len(txt.strip()) == 0:
                continue
            # 分支：如果是tokenizers.Tokenizer对象，调用encode
            if isinstance(tokenizer, HFTokenizer):
                enc = tokenizer.encode(txt)
                ids = enc.ids
            else:
                # 如果是transformers AutoTokenizer
                enc = tokenizer(txt, add_special_tokens=False)
                ids = enc["input_ids"]
            if add_special_tokens:
                # 可选：添加bos/eos tokens - 这里为了因果LM更简单的流程而跳过
                pass
            token_ids.extend(ids)

        # 分割成不重叠的块
        total_len = (len(token_ids) // block_size) * block_size
        self.blocks = []
        for i in range(0, total_len, block_size):
            chunk = token_ids[i:i+block_size]
            self.blocks.append(torch.tensor(chunk, dtype=torch.long))
        
        # 获取词汇表大小
        self.vocab_size = tokenizer.get_vocab_size() if isinstance(tokenizer, HFTokenizer) else len(tokenizer.get_vocab())

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]

# -----------------------------
# 简单的类GPT模型，带调整大小方法
# -----------------------------
class GPTLike(nn.Module):
    """类GPT模型"""
    def __init__(self, vocab_size=30000, block_size=128, n_layer=6, n_head=8, d_model=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.pos_emb = nn.Embedding(block_size, d_model)  # 位置嵌入层
        self.drop = nn.Dropout(dropout)  # Dropout层
        # Transformer块堆叠
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head, mlp_ratio=4.0, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)  # 最终层归一化
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 输出层
        # 权重绑定（输出层与嵌入层共享权重）
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights()  # 初始化权重

    def _init_weights(self):
        """初始化权重（GPT风格）"""
        for module in self.modules():
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
        device = idx.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # 位置索引
        x = self.tok_emb(idx) + self.pos_emb(pos)  # 词嵌入 + 位置嵌入
        x = self.drop(x)
        # 通过所有Transformer块
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)  # 最终层归一化
        logits = self.lm_head(x)  # 输出logits
        return logits  # (B, L, V)

    def resize_token_embeddings(self, new_vocab_size: int):
        """
        扩展或缩小token嵌入和lm_head到新的词汇表大小
        尽可能保留现有权重
        
        参数:
            new_vocab_size: 新的词汇表大小
        """
        old_weight = self.tok_emb.weight.data
        old_vocab, emb_dim = old_weight.shape
        if new_vocab_size == old_vocab:
            return self

        # 创建新嵌入层
        new_emb = nn.Embedding(new_vocab_size, emb_dim)
        nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)

        # 复制现有权重
        n_copy = min(old_vocab, new_vocab_size)
        new_emb.weight.data[:n_copy].copy_(old_weight[:n_copy])

        # 替换并分配
        self.tok_emb = new_emb
        # 重新创建lm_head并绑定权重
        new_lm_head = nn.Linear(emb_dim, new_vocab_size, bias=False)
        # 初始化然后复制
        nn.init.normal_(new_lm_head.weight, mean=0.0, std=0.02)
        new_lm_head.weight.data[:n_copy].copy_(old_weight[:n_copy])
        self.lm_head = new_lm_head
        # 权重绑定
        self.lm_head.weight = self.tok_emb.weight
        self.vocab_size = new_vocab_size
        return self

# Transformer block used above (same as earlier)
class CausalSelfAttention(nn.Module):
    """因果自注意力机制（防止未来信息泄露）"""
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)  # 多头注意力
        self.dropout = nn.Dropout(resid_dropout)  # 残差dropout
    
    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        # 创建因果掩码：上三角矩阵（不包括对角线）
        attn_mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)  # 带掩码的多头注意力
        return self.dropout(out)

class FeedForward(nn.Module):
    """前馈神经网络（两层线性变换+GELU激活）"""
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),  # GELU激活函数
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
        self.attn = CausalSelfAttention(d_model, nhead, resid_dropout=dropout)  # 自注意力
        self.ln2 = nn.LayerNorm(d_model)  # 层归一化2
        hidden = int(d_model * mlp_ratio)  # 前馈网络隐藏层维度
        self.mlp = FeedForward(d_model, hidden, dropout=dropout)  # 前馈网络
    
    def forward(self, x):
        # 残差连接+自注意力
        x = x + self.attn(self.ln1(x))
        # 残差连接+前馈网络
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------------------
# 数据准备和数据加载器
# -----------------------------
def prepare_dataloader_with_tokenizer(tokenizer, train_tokenizer_mode: bool, block_size: int, per_device_batch_size: int, rank: int, world_size: int):
    """准备数据加载器"""
    # 加载原始wikitext文本到内存（小数据集）
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex["text"] for ex in ds]
    # 构建分词后的数据集
    dataset = TokenizedBlocksDataset(tokenizer, texts, block_size=block_size, add_special_tokens=False)

    # 创建分布式采样器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=per_device_batch_size, sampler=sampler, drop_last=True, num_workers=2, pin_memory=True)
    return dataloader, dataset.vocab_size, sampler

# -----------------------------
# 训练循环（DDP）
# -----------------------------
def save_checkpoint(state, save_dir, step, is_main):
    """保存模型检查点（仅在主进程执行）"""
    if not is_main:
        return
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"ckpt_step{step}.pt")
    torch.save(state, fname)
    print(f"[rank 0] saved {fname}")

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args):
    """主训练函数"""
    # 初始化分布式环境
    rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    
    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 判断是否是主进程（rank=0）
    is_main = (rank == 0)

    if is_main:
        print("DDP GPT + BPE tokenizer demo")
        print("args:", args)

    # 设置随机种子（注意：每个rank使用不同的种子）
    set_seed(args.seed + rank)

    # --- tokenizer加载或训练 ---
    tokenizer = None
    if args.use_pretrained_tokenizer:
        # 使用transformers AutoTokenizer加载预训练分词器
        if is_main:
            print(f"[rank 0] loading pretrained tokenizer: {args.use_pretrained_tokenizer}")
        hf_tok = AutoTokenizer.from_pretrained(args.use_pretrained_tokenizer, use_fast=True)
        # 确保pad token存在
        if hf_tok.pad_token is None:
            hf_tok.add_special_tokens({"pad_token": "<|pad|>"})
        # 我们将使用transformers tokenizer API进行编码，但它背后是tokenizers rust实现
        tokenizer = hf_tok
        vocab_size = len(hf_tok)
        if is_main:
            print(f"[rank 0] loaded pretrained tokenizer vocab_size={vocab_size}")
    else:
        # 要么加载已训练的分词器json，要么训练新的（只在rank0上运行训练以避免竞争）
        if args.train_tokenizer and is_main and (not os.path.exists(args.tokenizer_path)):
            print("[rank 0] training BPE tokenizer on Wikitext (this may take a while)...")
            texts_iter = wikitext_texts_iterator()
            tk = train_bpe_tokenizer_on_texts(texts_iter, vocab_size=args.vocab_size, save_path=args.tokenizer_path)
            # tk是tokenizers.Tokenizer实例
            tokenizer = tk
            if is_main:
                print(f"[rank 0] tokenizer trained and saved to {args.tokenizer_path}")
        else:
            # 等待rank0完成训练文件写入
            if is_main:
                print(f"[rank 0] loading tokenizer from {args.tokenizer_path}")
            dist.barrier()  # 进程同步
            tokenizer = load_tokenizer_from_path(args.tokenizer_path)
            if is_main:
                print(f"[rank 0] loaded tokenizer vocab_size={tokenizer.get_vocab_size()}")

    # barrier确保分词器文件存在/所有rank有相同的视图
    dist.barrier()

    # 准备数据加载器
    dataloader, vocab_size, sampler = prepare_dataloader_with_tokenizer(tokenizer, args.train_tokenizer, args.block_size, args.per_device_batch_size, rank, world_size)

    if is_main:
        print(f"[rank 0] dataset blocks: {len(dataloader.dataset)}, vocab_size used={vocab_size}")

    # 使用初始vocab_size构建模型（如果tokenizers添加了tokens可能会调整）
    model = GPTLike(vocab_size=vocab_size, block_size=args.block_size, n_layer=6, n_head=8, d_model=512, dropout=0.1)
    # 如果我们使用了transformers AutoTokenizer并添加了pad token，需要调整大小
    # 但上面我们已经使用了数据集返回的vocab_size

    model.to(device)
    # 使用DDP包装模型[6](@ref)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

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

    # 训练循环
    global_step = 0
    model.train()
    start_time = time.time()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 关键：设置采样器的epoch[6](@ref)
        if is_main:
            print(f"Epoch {epoch+1}/{args.epochs} (steps {len(dataloader)})")
        
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
                elapsed = time.time() - start_time
                print(f"[step {global_step}] loss={loss_item:.4f} avg_epoch_loss={epoch_loss/(batch_idx+1):.4f} elapsed={elapsed:.1f}s")

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
        print("Training complete.")
    
    # 清理分布式资源
    cleanup_distributed()

# -----------------------------
# 程序入口
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
