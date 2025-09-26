#!/usr/bin/env python3
"""
简化的 DeepSeek-like 模型训练脚本（无分布式、无混合精度）
- 使用 BPE 分词器与 wikitext-2-raw-v1 数据集
- 使用 RoPE 位置编码（基于 cos/sin）替换原有位置嵌入
- 使用 MLA 代替 MHA，使用更高效的稀疏 MoE 代替 FFN
- 包含基本训练循环与检查点保存
启动示例:
    python deepseek_simple_train_moe_sparse.py --epochs 3 --batch_size 16
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

# 配置日志（可通过环境变量 LOG_LEVEL 调整）
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


def train_bpe_tokenizer(texts, vocab_size=30000, save_path="bpe_tokenizer.json"):
    """
    训练 BPE 分词器（逐条提供文本给 train_from_iterator）。
    注意：train_from_iterator 期望的是一个可迭代的文本序列（每次返回一个字符串）。
    """
    logger.info(f"训练 BPE 分词器: vocab_size={vocab_size}, save_path={save_path}")
    try:
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
            show_progress=True
        )

        # 传入逐条文本的 iterator（而不是批列表）
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
# 数据集：批量编码并按块组织
# -----------------------------
class TokenizedDataset(Dataset):
    """将所有文本批量编码为 token ids 并按 block_size 切块供训练使用。"""

    def __init__(self, texts, tokenizer: Tokenizer, block_size=256, batch_limit=1024):
        super().__init__()
        self.block_size = block_size

        logger.info(f"批量编码文本 (batch_limit={batch_limit})...")
        all_ids = []
        batch = []
        # 使用 tokenizer.encode_batch 分批编码以提高速度
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

        # 保证总长度可整除 block_size
        total_len = (len(all_ids) // block_size) * block_size
        if total_len == 0:
            raise ValueError(f"分词后数据不足以构成 block_size={block_size} 的块，请减小 block_size 或提供更多文本")

        arr = torch.tensor(all_ids[:total_len], dtype=torch.long)
        self.data = arr.view(-1, block_size)
        logger.info(f"TokenizedDataset: 总 token 数={len(all_ids)}, 块数={len(self.data)}")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        block = self.data[idx]
        x = block[:-1].clone()  # 输入序列 (长度 block_size-1)
        y = block[1:].clone()   # 目标序列（输入右移一位）
        return x, y


# -----------------------------
# RoPE 实现（基于 cos/sin 的常用实现）
# -----------------------------
def precompute_cos_sin(head_dim: int, max_seq_len: int, theta: float = 10000.0, device: torch.device = torch.device("cpu")):
    """
    预计算 RoPE 的 cos 与 sin 矩阵。
    返回 (cos, sin)，它们的形状为 (max_seq_len, head_dim//2)。
    head_dim 应为注意力头的维度（embedding_dim / num_heads）。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # (L, head_dim//2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin  # cos/sin 形状均为 (L, head_dim//2)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    将 RoPE 应用到 q 和 k 上（返回新的 tensor）。
    q, k 的形状为 (B, H, L, head_dim)
    cos, sin 的形状为 (L, head_dim//2)
    """
    B, H, L, head_dim = q.shape
    cos = cos[:L].unsqueeze(0).unsqueeze(0)  # (1,1,L,hd2)
    sin = sin[:L].unsqueeze(0).unsqueeze(0)

    # 偶数/奇数分量
    q_even = q[..., ::2]  # (B,H,L,hd2)
    q_odd = q[..., 1::2]
    k_even = k[..., ::2]
    k_odd = k[..., 1::2]

    # 旋转变换
    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd = q_even * sin + q_odd * cos
    k_rot_even = k_even * cos - k_odd * sin
    k_rot_odd = k_even * sin + k_odd * cos

    # 交错回去
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_out[..., ::2] = q_rot_even
    q_out[..., 1::2] = q_rot_odd
    k_out[..., ::2] = k_rot_even
    k_out[..., 1::2] = k_rot_odd
    return q_out, k_out


# -----------------------------
# 模型定义：MLA / MoE（稀疏）/ TransformerBlock / DeepSeekLike
# -----------------------------
class CausalMLA(nn.Module):
    """
    因果 Multi-Head Latent Attention (MLA)：
    - 通过对每个头的 KV 做低秩压缩以减小计算量/存储（示意实现）
    """

    def __init__(self, embed_dim, num_heads, latent_dim=None, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_dim = latent_dim if latent_dim is not None else max(1, self.head_dim // 4)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.q_compress = nn.Linear(self.head_dim, self.latent_dim)
        self.k_compress = nn.Linear(self.head_dim, self.latent_dim)
        self.v_compress = nn.Linear(self.head_dim, self.latent_dim)
        self.decompress = nn.Linear(self.latent_dim, self.head_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(resid_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin)  # 应用 RoPE

        q_compressed = self.q_compress(q)  # (B,H,L,latent_dim)
        k_compressed = self.k_compress(k)
        v_compressed = self.v_compress(v)

        attn_scores = torch.matmul(q_compressed, k_compressed.transpose(-2, -1)) / math.sqrt(self.latent_dim)

        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, L, L)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_out = torch.matmul(attn_probs, v_compressed)  # (B,H,L,latent_dim)
        attn_out = self.decompress(attn_out)  # (B,H,L,head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(attn_out)
        return self.dropout(out)


class Expert(nn.Module):
    """单个专家网络（简单的两层 FFN）。"""

    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x 可以是 (N, D) 或 (..., D)，线性会对最后一维生效
        return self.net(x)


class MoEFeedForward(nn.Module):
    """
    稀疏 Mixture of Experts（改进版）：
    - 仅对被选中的 token 调用对应专家（通过 gather/散回 scatter_add）
    - shared_experts 对所有 token 计算（因为它们总是激活）
    - top_k 允许每个 token 选择多个专家并按概率加权
    注意：该实现为教学友好型稀疏实现，使用 gather/索引来避免对所有专家做全量计算。
    """

    def __init__(self, d_model, hidden_dim, num_experts=8, top_k=2, num_shared=2, dropout=0.0):
        super().__init__()
        assert num_experts >= 1 and top_k >= 1 and top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared = max(0, num_shared)
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, hidden_dim, dropout) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList([Expert(d_model, hidden_dim, dropout) for _ in range(self.num_shared)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入:
            x: (B, L, D)
        输出:
            out: (B, L, D)
        稀疏流程:
            1) 计算 router logits -> top_k indices + top_k probs
            2) 对所有 token 计算 shared_experts（如果有）
            3) 对于每个 top-k（k=0..top_k-1），对每个被选专家 e:
                 - 找到被选的位置 idxs
                 - gather 输入 x_flat[idxs]（形状 (N_e, D)）
                 - 运行 expert_e(gathered) => out_e (N_e, D)
                 - moe_out[idxs] += out_e * gate_prob[idxs, k][:, None]
            4) 合并 shared_out 与 moe_out，reshape 回 (B,L,D)
        """
        device = x.device
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # (B*L, D)

        # 共享专家输出（在扁平化的输入上计算）
        if self.num_shared > 0:
            shared_out = sum(ex(x_flat) for ex in self.shared_experts) / max(1, self.num_shared)
        else:
            shared_out = torch.zeros_like(x_flat, device=device)

        # 路由：计算 top_k 专家及其概率
        gates = self.router(x_flat)  # (B*L, num_experts)
        topk_vals, topk_idx = torch.topk(gates, self.top_k, dim=-1)  # (B*L, top_k)
        topk_probs = nn.functional.softmax(topk_vals, dim=-1)  # (B*L, top_k)

        # 初始化 MoE 输出（扁平化）
        moe_out = torch.zeros_like(x_flat, device=device)  # (B*L, D)

        # 对每个 k，从对应被选专家处收集输入并计算
        # 这样我们只对被选位置调用对应专家
        for k_pos in range(self.top_k):
            # indices of selected experts for this k (shape: (B*L,))
            selected_experts = topk_idx[:, k_pos]  # 每个 token 在第 k_pos 个选择的专家 id
            gate_k = topk_probs[:, k_pos]  # (B*L,)

            # 对每个专家 e，找到被选的位置并处理
            # 这种循环的开销是 num_experts * top_k，但实际 expert 计算只针对被选位置
            for e in range(self.num_experts):
                # 找到在第 k_pos 被选为 e 的 token 索引
                mask = (selected_experts == e)
                if not mask.any():
                    continue
                idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)  # 一维索引 (N_e,)
                # gather inputs
                inp = x_flat.index_select(0, idxs)  # (N_e, D)
                # 运行对应专家（仅对这些位置）
                out_e = self.experts[e](inp)  # (N_e, D)
                # 权重（gate_k）对应位置
                w = gate_k.index_select(0, idxs).unsqueeze(1)  # (N_e,1)
                # 将结果按位置累加（加到 moe_out 对应位置）
                moe_out.index_add_(0, idxs, out_e * w)

        # 合并共享专家输出与 MoE 输出，并恢复形状 (B,L,D)
        out = (shared_out + moe_out).view(B, L, D)
        return self.dropout(out)


class TransformerBlock(nn.Module):
    """Transformer 块：LayerNorm -> MLA -> LayerNorm -> MoEFFN（均带残差）。"""

    def __init__(self, d_model, nhead, mlp_ratio=4.0, dropout=0.1, latent_dim=None, num_experts=8, top_k=2, num_shared=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalMLA(d_model, nhead, latent_dim=latent_dim, attn_dropout=0.0, resid_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = MoEFeedForward(d_model, hidden, num_experts=num_experts, top_k=top_k, num_shared=num_shared, dropout=dropout)

    def forward(self, x, cos: torch.Tensor, sin: torch.Tensor):
        # 注意：attn 需要 cos, sin（RoPE 预计算）
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class DeepSeekLike(nn.Module):
    """整体模型，使用 MLA + 稀疏 MoE + RoPE。"""

    def __init__(self, vocab_size=30000, block_size=256, n_layer=6, n_head=8, d_model=768, dropout=0.1,
                 latent_dim=None, num_experts=8, top_k=2, num_shared=2, rope_theta=10000.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, mlp_ratio=4.0, dropout=dropout,
                             latent_dim=latent_dim, num_experts=num_experts, top_k=top_k, num_shared=num_shared)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight
        self.block_size = block_size
        self.d_model = d_model
        self.n_head = n_head
        self.rope_theta = rope_theta
        self.apply(self._init_weights)

        # 预计算 RoPE 的 cos/sin（head_dim = d_model // n_head）
        head_dim = d_model // n_head
        # 这里先保存在 cpu 上，forward 时再移动到目标 device（节省显存）
        self.cos, self.sin = precompute_cos_sin(head_dim, block_size, theta=rope_theta, device=torch.device("cpu"))

    def _init_weights(self, module):
        """初始化权重：线性和嵌入用正态分布，LayerNorm 用常规初始化。"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx: torch.LongTensor):
        """
        前向：
        - idx: (B, L) token ids
        - 返回 logits: (B, L, vocab_size)
        """
        B, L = idx.size()
        if L > self.block_size:
            idx = idx[:, :self.block_size]
            L = self.block_size

        x = self.tok_emb(idx)  # (B, L, D)
        x = self.drop(x)

        # 将 cos/sin 移动到当前 device 并截断到序列长度
        cos = self.cos.to(idx.device)
        sin = self.sin.to(idx.device)

        for blk in self.blocks:
            x = blk(x, cos, sin)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# -----------------------------
# 主训练流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="简化的 DeepSeek-like 模型训练（稀疏 MoE 版本）")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--block_size", type=int, default=256, help="序列块大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--vocab_size", type=int, default=30000, help="分词器词汇大小（为训练 tokenizer 时使用）")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=768, help="模型隐藏维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")
    parser.add_argument("--save_interval", type=int, default=1, help="保存检查点间隔（轮）")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="梯度裁剪阈值 (<=0 表示不裁剪)")
    # DeepSeek-like 参数
    parser.add_argument("--latent_dim", type=int, default=None, help="MLA 中的 latent 维度（默认 head_dim//4）")
    parser.add_argument("--num_experts", type=int, default=8, help="MoE 专家数量")
    parser.add_argument("--top_k", type=int, default=2, help="MoE top-k")
    parser.add_argument("--num_shared", type=int, default=2, help="MoE 共享专家数量")
    # RoPE 参数
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE 的 theta 参数")
    args = parser.parse_args()

    # 参数检查
    if args.block_size <= 0:
        raise ValueError("block_size 必须大于 0")
    if args.vocab_size <= 0:
        raise ValueError("vocab_size 必须大于 0")
    if args.d_model % args.n_head != 0:
        raise ValueError("d_model 必须能被 n_head 整除")

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据与 tokenizer
    texts = prepare_data()
    tokenizer_path = "bpe_tokenizer.json"
    if not Path(tokenizer_path).exists():
        tokenizer = train_bpe_tokenizer(texts, vocab_size=args.vocab_size, save_path=tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        logger.info(f"从 {tokenizer_path} 加载分词器")

    # 实际 vocab size（以 tokenizer 为准）
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"分词器词汇大小: {vocab_size}")

    # 数据集与 dataloader
    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size, batch_limit=1024)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=min(4, max(0, multiprocessing.cpu_count() - 1)),
        drop_last=True
    )

    # 模型初始化并移动到 device
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

    # 优化器、损失与调度器
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

                # 前向
                logits = model(x)  # (B, L, V)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))

                # 反向
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪（可选）
                if args.clip_grad_norm and args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                optimizer.step()
                scheduler.step()

                loss_val = float(loss.item())
                total_loss += loss_val
                num_batches += 1

                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss_val:.4f}")

            # epoch 汇总与 checkpoint 保存
            avg_loss = total_loss / max(1, num_batches)
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr
            logger.info(f"Epoch {epoch} 完成. 平均损失={avg_loss:.4f}, 当前学习率={lr_now:.6g}")

            # 保存检查点
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

                # 可选：保留最近 5 个检查点（按 epoch 简化）
                max_keep = 5
                if epoch // args.save_interval > max_keep:
                    old_epoch = epoch - args.save_interval * max_keep
                    old_ckpt = save_dir / f"model_epoch_{old_epoch}.pth"
                    if old_ckpt.exists():
                        old_ckpt.unlink()
                        logger.info(f"删除旧检查点: {old_ckpt}")

    except RuntimeError as e:
        logger.exception(f"训练失败（RuntimeError）: {str(e)}")
        raise
    except ValueError as e:
        logger.exception(f"参数错误（ValueError）: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"未知错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
