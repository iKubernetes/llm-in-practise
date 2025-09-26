#!/usr/bin/env python3
"""
简化的 GPT-like 模型训练脚本（无分布式训练、无混合精度）
- 使用 BERT 的 WordPiece 分词器（bert-base-uncased 预训练词表，大小 30522）
- 使用 wikitext-2-raw-v1 数据集
- 包含基本的模型训练和检查点保存逻辑
启动示例:
    python GPTLike_wikitext2_bert_tokenizer.py --epochs 3 --batch_size 16
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
from transformers import BertTokenizer
from datasets import load_dataset

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

def load_bert_tokenizer(model_name="bert-base-uncased"):
    """加载 BERT 的预训练 WordPiece 分词器。"""
    logger.info(f"加载 BERT 分词器: {model_name}")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        vocab_size = len(tokenizer.vocab)
        logger.info(f"BERT 分词器词汇大小: {vocab_size}")
        return tokenizer, vocab_size
    except Exception:
        logger.exception("BERT 分词器加载失败")
        raise

# -----------------------------
# 数据集：批量编码与块视图
# -----------------------------
class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer: BertTokenizer, block_size=256, batch_limit=1024):
        """初始化数据集，批量编码文本并按块组织。"""
        super().__init__()
        self.block_size = block_size
        self.tokenizer = tokenizer

        logger.info(f"批量编码文本 (batch_limit={batch_limit})...")
        all_ids = []
        batch = []
        for t in texts:
            batch.append(t)
            if len(batch) >= batch_limit:
                # 使用 BERT 分词器编码，添加特殊 token 并限制长度
                encodings = tokenizer(
                    batch,
                    add_special_tokens=False,  # 不添加 [CLS] 和 [SEP]
                    truncation=True,
                    max_length=block_size,
                    return_tensors=None
                )
                for enc in encodings['input_ids']:
                    all_ids.extend(enc)
                batch = []
        if batch:
            encodings = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=True,
                max_length=block_size,
                return_tensors=None
            )
            for enc in encodings['input_ids']:
                all_ids.extend(enc)

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
    def __init__(self, vocab_size=30522, block_size=256, n_layer=6, n_head=8, d_model=768, dropout=0.1):
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
    parser = argparse.ArgumentParser(description="简化的 GPT-like 模型训练（使用 BERT 分词器）")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--block_size", type=int, default=256, help="序列块大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=768, help="模型维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")
    parser.add_argument("--save_interval", type=int, default=1, help="检查点保存间隔")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    args = parser.parse_args()

    # 参数验证
    if args.block_size <= 0:
        raise ValueError("block_size 必须大于 0")

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

    # 加载数据和 BERT 分词器
    texts = prepare_data()
    tokenizer, vocab_size = load_bert_tokenizer("bert-base-uncased")

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
    model = GPTLike(
        vocab_size=vocab_size,  # 使用 BERT 分词器的词汇表大小
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout
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
                scheduler.step()

                loss_val = float(loss.item())
                total_loss += loss_val
                num_batches += 1

                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch} Batch {batch_idx} Loss {loss_val:.4f}")

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

if __name__ == "__main__":
    main()
