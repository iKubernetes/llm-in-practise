#!/usr/bin/env python3
"""
简化的 DeepSeek-like 模型训练脚本（无分布式训练、无混合精度）
- 基于 GPT-like 模型修改，使用 MLA 替换 MHA，使用 MoE 替换 FFN
- 使用 RoPE 位置编码替换原有位置嵌入
- 使用 BPE 分词器和 wikitext-2-raw-v1 数据集
- 包含基本的模型训练和检查点保存逻辑
启动示例:
    python deepseek_simple_train.py --epochs 3 --batch_size 16
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

# 第三方库：用于分词和数据集加载
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

# 禁用 tokenizers 并行性以避免 fork 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# 数据加载与分词
# -----------------------------
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

        # FIXED: train_from_iterator expects an iterator of **strings**, not batches (lists)
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
# RoPE 辅助函数
# -----------------------------
def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0, device: torch.device = torch.device("cpu")):
    """
    预计算 RoPE 的频率和复数表示（cos + i sin）。
    返回形状 (max_seq_len, dim/2) 的 complex 张量（complex64）。
    """
    # dim 是 head_dim（必须为偶数以配对 real/imag）
    if dim % 2 != 0:
        raise ValueError(f"head_dim (dim) must be even for RoPE, got {dim}")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs).float()  # (max_seq_len, dim/2)
    # polar(magnitude, angle) -> complex tensor
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex tensor
    return freqs_cis  # complex dtype

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
    """
    应用 RoPE 到查询和键向量。
    q, k: (B, H, L, head_dim)
    freqs_cis: complex tensor (max_seq_len, head_dim//2)
    返回 q_out, k_out 与原始 dtype。
    """
    # q, k must have last dim even
    B, H, L, head_dim = q.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for rotary embeddings, got {head_dim}")

    # Convert q,k to complex by pairing last-dim elements (real, imag)
    # reshape -> (..., head_dim//2, 2) then view_as_complex -> complex with last dim head_dim//2
    q_ = torch.view_as_complex(q.float().reshape(B, H, L, head_dim // 2, 2))
    k_ = torch.view_as_complex(k.float().reshape(B, H, L, head_dim // 2, 2))

    # slice freqs_cis to length L and broadcast to (1,1,L,head_dim//2)
    freq = freqs_cis[:L].unsqueeze(0).unsqueeze(0)  # (1,1,L,head_dim//2) complex
    # multiply in complex domain
    q_out_c = q_ * freq
    k_out_c = k_ * freq

    # convert back to real interleaved format and reshape to original last dim
    q_out = torch.view_as_real(q_out_c).reshape(B, H, L, head_dim)
    k_out = torch.view_as_real(k_out_c).reshape(B, H, L, head_dim)
    return q_out.type_as(q), k_out.type_as(k)

# -----------------------------
# 模型定义 (DeepSeek-like with RoPE)
# -----------------------------
class CausalMLA(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim=None, attn_dropout=0.0, resid_dropout=0.0):
        """
        初始化因果 Multi-Head Latent Attention (MLA) 模块。
        q,k,v shapes internally: (B, H, L, head_dim)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads
        # FIXED: ensure head_dim even for RoPE
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even (required for RoPE). Please set d_model/n_head to an even head_dim.")
        # latent_dim default and min 1
        self.latent_dim = max(1, latent_dim if latent_dim is not None else max(1, self.head_dim // 4))

        # 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # FIXED: Linear layers operate on last dim; don't transpose dims incorrectly
        # compress/decompress are applied to last dimension (head_dim <-> latent_dim)
        self.q_compress = nn.Linear(self.head_dim, self.latent_dim)
        self.k_compress = nn.Linear(self.head_dim, self.latent_dim)
        self.v_compress = nn.Linear(self.head_dim, self.latent_dim)
        self.decompress = nn.Linear(self.latent_dim, self.head_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(resid_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, freqs_cis: torch.Tensor):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)  # still (B,H,L,head_dim)

        # FIXED: compress along last dim (head_dim)
        # Linear accepts (..., in_features) -> (..., out_features), so we can call directly
        q_compressed = self.q_compress(q)  # (B,H,L,latent_dim)
        k_compressed = self.k_compress(k)
        v_compressed = self.v_compress(v)

        # attention scores and mask
        # compute matmul on last two dims: (..., L, latent_dim) x (..., latent_dim, L) => (..., L, L)
        attn_scores = torch.matmul(q_compressed, k_compressed.transpose(-2, -1)) / math.sqrt(max(1, self.latent_dim))

        # causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, L, L)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # attention output using compressed V
        attn_out = torch.matmul(attn_probs, v_compressed)  # (B,H,L,latent_dim)

        # decompress last dim back to head_dim
        attn_out = self.decompress(attn_out)  # (B,H,L,head_dim)

        # merge heads back to (B,L,D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(attn_out)
        return self.dropout(out)

class Expert(nn.Module):
    """单个专家网络（类似于 FFN）。"""
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

class MoEFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts=8, top_k=2, num_shared=2, dropout=0.0):
        """初始化 Mixture of Experts (MoE) FFN。"""
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared = num_shared
        self.router = nn.Linear(d_model, num_experts)  # 路由器
        self.experts = nn.ModuleList([Expert(d_model, hidden_dim, dropout) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList([Expert(d_model, hidden_dim, dropout) for _ in range(num_shared)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # (B*L, D)

        # FIXED: shared_out should use x_flat
        if self.num_shared > 0:
            shared_out = sum(expert(x_flat) for expert in self.shared_experts) / max(1, self.num_shared)
        else:
            shared_out = torch.zeros_like(x_flat)

        # routing
        gates = self.router(x_flat)  # (B*L, num_experts)
        top_k_gates, top_k_indices = torch.topk(gates, min(self.top_k, self.num_experts), dim=-1)  # (B*L, top_k)
        top_k_gates = nn.functional.softmax(top_k_gates, dim=-1)

        moe_out = torch.zeros_like(x_flat)

        # For each chosen top-k, compute expert outputs where mask indicates selection
        # This is a simple (but not most efficient) implementation
        for i in range(top_k_indices.size(1)):
            indices_i = top_k_indices[:, i]  # (B*L,)
            weights_i = top_k_gates[:, i].unsqueeze(1)  # (B*L,1)

            # Create one-hot mask per expert (long->float)
            # We'll compute each expert output only for tokens assigned to it
            # Build a mask matrix to select tokens for each expert (could be memory heavy if many experts)
            mask = torch.zeros_like(gates, dtype=torch.bool)
            mask.scatter_(1, indices_i.unsqueeze(1), True)  # (B*L, num_experts)

            # Accumulate outputs from all experts for this top-k slot
            expert_out = torch.zeros_like(x_flat)
            for e in range(self.num_experts):
                sel = mask[:, e]
                if sel.any():
                    inp = x_flat[sel]  # (n_sel, D)
                    out_e = self.experts[e](inp)  # (n_sel, D)
                    # scatter back
                    expert_out[sel] = out_e

            # weight by the gate
            moe_out += expert_out * weights_i

        out = (shared_out + moe_out).view(B, L, D)
        return self.dropout(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1, latent_dim=None, num_experts=8, top_k=2, num_shared=2):
        """初始化 Transformer 块（使用 MLA 和 MoE FFN）。"""
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalMLA(d_model, nhead, latent_dim=latent_dim, attn_dropout=0.0, resid_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = MoEFeedForward(d_model, hidden, num_experts=num_experts, top_k=top_k, num_shared=num_shared, dropout=dropout)

    def forward(self, x, freqs_cis: torch.Tensor):
        x = x + self.attn(self.ln1(x), freqs_cis)  # 残差连接：注意力（传递 freqs_cis）
        x = x + self.mlp(self.ln2(x))   # 残差连接：MoE FFN
        return x

class DeepSeekLike(nn.Module):
    def __init__(self, vocab_size=30000, block_size=256, n_layer=6, n_head=8, d_model=768, dropout=0.1,
                 latent_dim=None, num_experts=8, top_k=2, num_shared=2, rope_theta=10000.0):
        """初始化 DeepSeek-like 模型，使用 MLA、MoE 和 RoPE。"""
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, mlp_ratio=4.0, dropout=dropout,
                             latent_dim=latent_dim, num_experts=num_experts, top_k=top_k, num_shared=num_shared)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # 共享输入和输出嵌入权重
        self.head.weight = self.tok_emb.weight
        self.block_size = block_size
        self.d_model = d_model
        self.n_head = n_head
        self.rope_theta = rope_theta
        self.apply(self._init_weights)

        # 预计算 RoPE 频率（最大序列长度为 block_size）
        head_dim = d_model // n_head
        # FIXED: precompute requires head_dim even; check earlier already
        self.freqs_cis = precompute_freqs_cis(head_dim, block_size, theta=rope_theta)

    def _init_weights(self, module):
        """初始化权重：线性层和嵌入层使用正态分布，LayerNorm 初始化为标准值。"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
        x = self.tok_emb(idx)
        x = self.drop(x)
        freqs_cis = self.freqs_cis.to(idx.device)  # 移动到当前设备
        for blk in self.blocks:
            x = blk(x, freqs_cis)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -----------------------------
# 主训练流程
# -----------------------------
def main():
    """主函数：解析参数、设置环境、训练模型并保存检查点。"""
    parser = argparse.ArgumentParser(description="简化的 DeepSeek-like 模型训练")
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
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    # DeepSeek-like 参数
    parser.add_argument("--latent_dim", type=int, default=None, help="MLA 中的 latent 维度（默认 head_dim // 4）")
    parser.add_argument("--num_experts", type=int, default=8, help="MoE 中的专用专家数量")
    parser.add_argument("--top_k", type=int, default=2, help="MoE 中选择的 top-k 专家")
    parser.add_argument("--num_shared", type=int, default=2, help="MoE 中的共享专家数量")
    # RoPE 参数
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE 的 theta 值")
    args = parser.parse_args()

    # 参数验证
    if args.block_size <= 0:
        raise ValueError("block_size 必须大于 0")
    if args.vocab_size <= 0:
        raise ValueError("vocab_size 必须大于 0")
    if args.d_model % args.n_head != 0:
        raise ValueError("d_model 必须能被 n_head 整除")
    # head_dim 必须为偶数以支持 RoPE
    if (args.d_model // args.n_head) % 2 != 0:
        raise ValueError("d_model/n_head (head_dim) 必须为偶数以支持 RoPE")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据和分词器
    texts = prepare_data()
    tokenizer_path = "bpe_tokenizer.json"
    if not Path(tokenizer_path).exists():
        tokenizer = train_bpe_tokenizer(
            texts,
            vocab_size=args.vocab_size,
            save_path=tokenizer_path,
            batch_size=1024
        )
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        logger.info(f"从 {tokenizer_path} 加载分词器")

    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"分词器词汇大小: {vocab_size}")

    # 创建数据集和数据加载器
    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size, batch_limit=1024)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=min(4, max(0, multiprocessing.cpu_count() - 1)),
        drop_last=True
    )

    # 初始化模型
    model = DeepSeekLike(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_shared=args.num_shared,
        rope_theta=args.rope_theta
    ).to(device)

    # 初始化优化器、损失函数和调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # 训练循环
    try:
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # 前向传播
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                # 优化步骤
                optimizer.step()

                loss_val = float(loss.item())
                total_loss += loss_val
                num_batches += 1

                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss_val:.4f}")

            # epoch-level scheduler step (FIXED: avoid stepping per-batch)
            scheduler.step()

            # epoch 总结与检查点保存
            avg_loss = total_loss / max(1, num_batches)
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr
            logger.info(f"Epoch {epoch} 完成. 平均损失={avg_loss:.4f}, 当前学习率={lr_now:.6g}")

            if epoch % args.save_interval == 0:
                ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "vocab_size": vocab_size,
                    "block_size": args.block_size,
                    "args": vars(args)
                }
                ckpt_path = save_dir / f"model_epoch_{epoch}.pth"
                torch.save(ckpt, ckpt_path)
                logger.info(f"检查点已保存至 {ckpt_path}")

                # 清理旧检查点（保留最近 5 个）
                # FIXED: 删除策略按 epoch 计数，而不是误删
                oldest_keep_epoch = max(0, epoch - args.save_interval * 5)
                old_ckpt = save_dir / f"model_epoch_{oldest_keep_epoch}.pth"
                if old_ckpt.exists() and oldest_keep_epoch < epoch:
                    try:
                        old_ckpt.unlink()
                        logger.info(f"删除旧检查点: {old_ckpt}")
                    except Exception:
                        logger.warning(f"无法删除旧检查点: {old_ckpt}")

    except RuntimeError as e:
        logger.exception(f"训练失败: {str(e)}")
        raise
    except ValueError as e:
        logger.exception(f"参数错误: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"未知错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
