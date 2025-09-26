#!/usr/bin/env python3
"""
核心功能:
- 通过命令行选择GPU数量 (默认使用所有GPU, 最少1个)
- 训练BPE分词器
- 自定义GPT-like Transformer模型
- DDP多GPU分布式训练
- 混合精度训练与模型检查点保存
- 增强错误处理和日志
- 支持WikiText-103-v1数据集
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
import argparse
import logging

# 禁用 tokenizers 并行性（推荐但非必须）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# 1. 分布式训练设置
# -----------------------------
def setup_distributed():
    """初始化分布式训练环境，使用 torchrun 提供的环境变量"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        try:
            dist.init_process_group(backend="nccl")
            logger.info(f"Rank {rank} 初始化完成, 世界大小: {world_size}")
        except Exception as e:
            logger.error(f"Rank {rank} 初始化失败: {e}")
            raise
    return rank, world_size

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("分布式环境清理完成")

# -----------------------------
# 2. 数据准备与 BPE 分词器
# -----------------------------
def prepare_data():
    """加载 WikiText-103-v1 数据集"""
    logger.info("加载 WikiText-103-v1 数据集...")
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        texts = [ex["text"] for ex in dataset if ex["text"].strip()]
        logger.info(f"加载 {len(texts)} 条文本")
        return texts
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        raise

def train_bpe_tokenizer(texts, vocab_size=3000, save_path="bpe_tokenizer.json"):
    """训练 BPE 分词器"""
    logger.info(f"训练 BPE 分词器, 词汇表大小: {vocab_size}...")
    try:
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
            show_progress=True
        )
        
        tokenizer.train_from_iterator(texts, trainer=trainer)
        tokenizer.save(save_path)
        logger.info(f"分词器训练完成，保存至 {save_path}")
        return tokenizer
    except Exception as e:
        logger.error(f"分词器训练失败: {e}")
        raise

# -----------------------------
# 3. 数据集处理 (分布式兼容)
# -----------------------------
class TokenizedDataset(Dataset):
    """处理分词后的数据集"""
    def __init__(self, texts, tokenizer, block_size=128):
        self.block_size = block_size
        self.data = []
        
        # 编码所有文本
        all_token_ids = []
        for text in texts:
            if text.strip():
                encoded = tokenizer.encode(text)
                all_token_ids.extend(encoded.ids)
        
        # 创建训练块
        total_len = len(all_token_ids) // block_size * block_size
        for i in range(0, total_len, block_size):
            self.data.append(all_token_ids[i:i + block_size])
        logger.info(f"数据集准备完成, 总块数: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx][:-1]  # 输入序列
        y = self.data[idx][1:]   # 目标序列
        return torch.tensor(x), torch.tensor(y)

# -----------------------------
# 4. Transformer 模型定义
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.dropout = nn.Dropout(resid_dropout)
    
    def forward(self, x):
        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
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

class GPTLike(nn.Module):
    def __init__(self, vocab_size=256, block_size=128, n_layer=6, n_head=8, d_model=512, dropout=0.1):
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
        self.head.weight = self.tok_emb.weight
        self.block_size = block_size
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
        pos = torch.arange(L, device=idx.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -----------------------------
# 5. 训练函数 (DDP 兼容)
# -----------------------------
def main():
    """主训练函数，兼容 torchrun"""
    parser = argparse.ArgumentParser(description="优化版 BPE + DDP 训练")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--block_size", type=int, default=512, help="序列长度")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--vocab_size", type=int, default=3000, help="词汇表大小")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 率")
    parser.add_argument("--save_interval", type=int, default=1, help="保存间隔")
    parser.add_argument("--num_gpus", type=int, default=None, help="使用 GPU 数量 (默认所有可用 GPU)")
    args = parser.parse_args()
    
    # 获取可用 GPU 数量
    total_gpus = torch.cuda.device_count()
    logger.info(f"检测到 {total_gpus} 个可用 GPU")
    
    # 确定使用 GPU 数量，强制与 --nproc_per_node 一致
    world_size = args.num_gpus if args.num_gpus is not None else total_gpus
    world_size = max(1, min(world_size, total_gpus))  # 限制在可用 GPU 内
    nproc_per_node = int(os.environ.get("NPROC_PER_NODE", world_size))  # 获取 torchrun 的进程数
    world_size = min(world_size, nproc_per_node)  # 使用 torchrun 指定的进程数
    logger.info(f"配置使用 {world_size} 个 GPU")
    
    # 单 GPU 或多 GPU 模式
    rank, effective_world_size = setup_distributed()
    if effective_world_size == 1:
        world_size = 1  # 强制单 GPU 模式
    
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # 设置随机种子
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    
    # 加载数据
    texts = prepare_data()
    
    # 训练或加载分词器
    tokenizer_path = "bpe_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        tokenizer = train_bpe_tokenizer(texts, vocab_size=args.vocab_size, save_path=tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # 获取实际词汇表大小
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"分词器词汇表大小: {vocab_size}")
    
    # 创建数据集和数据加载器
    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(world_size == 1),
        pin_memory=True,
        num_workers=2
    )
    
    # 创建模型
    model = GPTLike(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    
    # 训练循环
    for epoch in range(args.epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(x)
                loss = criterion(outputs.view(-1, vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if rank == 0 and batch_idx % 100 == 0 and world_size > 1:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            elif batch_idx % 100 == 0 and world_size == 1:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        if rank == 0 or world_size == 1:
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch} 完成, 平均损失: {avg_loss:.4f}")
            
            if epoch % args.save_interval == 0:
                checkpoint_path = f"data/ddp_gpt_bpe_wikitext103/model_epoch_{epoch}.pth"
                if world_size > 1:
                    torch.save(model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"模型保存至 {checkpoint_path}")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()
