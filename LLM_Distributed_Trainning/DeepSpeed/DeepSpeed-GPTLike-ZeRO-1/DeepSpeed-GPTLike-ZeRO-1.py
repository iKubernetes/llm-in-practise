# train_deepspeed.py
"""
DeepSpeed 包装的简化版 GPT-like 训练脚本（用于初学者学习）
说明：
- 使用独立的 ds_config.json 配置文件（见下方示例）
- 自动从 ds_config.json 读取 train_micro_batch_size_per_gpu 并用于 DataLoader 的 batch_size（若命令行同时传入 --batch_size，则以 ds_config 为准并打印提示）
- 支持 ZeRO Stage-1 与 混合精度（fp16）
- 使用中文注释，尽量保持脚本简洁，便于教学
- deepspeed 启动示例（在控制台运行）：
  deepspeed --num_gpus=4 train_deepspeed.py --epochs 3 --n_layer 6 --n_head 12 --d_model 768 --dropout 0.1 --ds_config ds_config.json --local_rank $LOCAL_RANK

注意：
- 如果使用 deepspeed launcher，通常不需要手动传入 --local_rank（launcher 会自动设置）。脚本仍保留该参数以满足题目要求。
- 本脚本会尽量将 DataLoader 的 batch_size 与 ds_config 中的 train_micro_batch_size_per_gpu 保持一致，减少手动同步错误。
- 修复了进程退出时未销毁 torch.distributed 进程组导致的 NCCL 警告（在退出前尝试优雅销毁进程组）。
"""

import os
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer
from datasets import load_dataset

# DeepSpeed
try:
    import deepspeed
except Exception:
    deepspeed = None

# 配置基础日志，所有进程都会创建 logger，但我们会根据 rank 控制输出
logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# 数据加载与分词
# -----------------------------
def prepare_data():
    """加载 wikitext-2-raw-v1 数据集的训练集，返回非空文本列表"""
    logger.info("加载 WikiText 数据集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [ex["text"] for ex in dataset if ex.get("text") and ex["text"].strip()]
    logger.info(f"加载 {len(texts)} 条非空文本")
    return texts

# -----------------------------
# 数据集
# -----------------------------
class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=256):
        """初始化数据集，按块组织编码后的 token，确保序列长度不超过 512"""
        self.block_size = block_size
        if block_size > 512:
            raise ValueError(f"block_size ({block_size}) 超过 BERT 最大序列长度 512")
        logger.info("编码文本...")
        all_ids = []
        for t in texts:
            token_ids = tokenizer.encode(t, add_special_tokens=False, max_length=512, truncation=True)
            all_ids.extend(token_ids)

        total_len = (len(all_ids) // block_size) * block_size
        if total_len == 0:
            raise ValueError(f"数据不足以构成 block_size={block_size} 的块")

        arr = torch.tensor(all_ids[:total_len], dtype=torch.long)
        self.data = arr.view(-1, block_size)
        logger.info(f"总 token 数={len(all_ids)}, 块数={len(self.data)}")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        block = self.data[idx]
        return block[:-1], block[1:]

# -----------------------------
# 模型定义（与原脚本一致）
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
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
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, int(d_model * 4), dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTLike(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, d_model, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.block_size = block_size

        # 固定正弦/余弦位置编码
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pe.unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
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

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :L, :].expand(B, -1, -1)
        x = self.drop(tok_emb + pos_emb)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

# -----------------------------
# 主训练流程（DeepSpeed 兼容）
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="简化的 GPT-like 模型训练（DeepSpeed 版）")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="（可选）本地 DataLoader 的批次大小；若 ds_config.json 中存在 train_micro_batch_size_per_gpu 则以该值为准")
    parser.add_argument("--block_size", type=int, default=256, help="序列块大小（不得超过 512）")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n_layer", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--n_head", type=int, default=12, help="注意力头数")
    parser.add_argument("--d_model", type=int, default=768, help="模型维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比率")
    # DeepSpeed 专用选项（按题目要求接收 --ds_config 与 --local_rank）
    parser.add_argument("--ds_config", type=str, default="ds_config.json", help="DeepSpeed 配置文件路径（JSON）")
    parser.add_argument("--local_rank", type=int, default=0, help="进程本地 rank（由 deepspeed launcher 提供）")
    args = parser.parse_args()

    # 基本参数检查
    if args.block_size > 512:
        raise ValueError(f"block_size ({args.block_size}) 超过 BERT 最大序列长度 512")
    if args.d_model % args.n_head != 0:
        raise ValueError(f"d_model ({args.d_model}) 必须能被 n_head ({args.n_head}) 整除")

    # 尝试读取 ds_config.json 中的 train_micro_batch_size_per_gpu
    ds_batch = None
    if args.ds_config and os.path.isfile(args.ds_config):
        try:
            with open(args.ds_config, 'r') as f:
                ds_conf = json.load(f)
            ds_batch = ds_conf.get('train_micro_batch_size_per_gpu')
            if ds_batch is not None:
                try:
                    ds_batch = int(ds_batch)
                except Exception:
                    logger.warning('从 ds_config.json 读取到 train_micro_batch_size_per_gpu，但无法转换为 int，忽略该值')
                    ds_batch = None
        except Exception as e:
            logger.warning(f'无法读取或解析 ds_config.json: {e}')

    # 优先使用 ds_config 中的 train_micro_batch_size_per_gpu 设置 DataLoader batch_size
    if ds_batch is not None:
        dataloader_batch_size = ds_batch
        logger.info(f"从 ds_config.json 读取到 train_micro_batch_size_per_gpu={ds_batch}，将用于 DataLoader.batch_size（覆盖命令行 --batch_size）")
    else:
        dataloader_batch_size = args.batch_size
        logger.info(f"未在 ds_config.json 中找到 train_micro_batch_size_per_gpu，使用命令行的 --batch_size={args.batch_size}")

    # 如果未安装 deepspeed，则提示并退出
    if deepspeed is None:
        logger.error("未检测到 deepspeed，请先安装 deepspeed: pip install deepspeed")
        return

    # deepspeed 会自动通过 launcher 设置环境变量；这里尝试初始化分布式
    try:
        deepspeed.init_distributed()
    except Exception:
        # 若 init_distributed 已在 launcher 中完成，这里可能抛异常，可以安全忽略
        pass

    # 获取全局 rank 与 world_size（若未初始化则默认为 0/1）
    world_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # 只有 rank 0 在控制台打印更多信息
    is_main_process = (world_rank == 0)
    if is_main_process:
        logger.info(f"DeepSpeed 环境：rank={world_rank}, world_size={world_size}")

    # 设置设备：优先使用 local_rank 对应的 GPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if is_main_process:
        logger.info(f"使用设备: {device}")

    # 加载数据与分词器
    texts = prepare_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    if is_main_process:
        logger.info(f"BERT 分词器词汇大小: {vocab_size}")

    # 数据集与 DataLoader：对于分布式训练建议使用 DistributedSampler，便于数据划分
    dataset = TokenizedDataset(texts, tokenizer, block_size=args.block_size)
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=dataloader_batch_size, sampler=sampler, shuffle=(sampler is None), drop_last=True, num_workers=4)

    # 初始化模型与优化器
    model = GPTLike(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout
    )

    # 将模型参数移动到默认设备（DeepSpeed 在 initialize 时会处理实际分配）
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # DeepSpeed 初始化：将 model/optimizer 包装为 engine
    # 注意：deepspeed.initialize 可以接受 args、model、optimizer、config_params/file
    # 这里传入 args 与 config 文件路径（args.ds_config）
    engine = None
    try:
        engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer, config=args.ds_config)

        # 创建保存目录（每个进程都创建不会冲突）
        Path("checkpoints").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

        # 训练循环（使用 engine 的 backward/step）
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)  # 保证各 epoch 数据随机性
            engine.train()
            total_loss, num_batches = 0.0, 0
            batch_losses = []

            for batch_idx, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)

                logits = engine(x)  # engine 会代理到模型
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))

                engine.backward(loss)
                engine.step()

                batch_losses.append(loss.item())
                total_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % 50 == 0 and is_main_process:
                    avg_loss = sum(batch_losses[-50:]) / min(len(batch_losses), 50)
                    logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}, Avg Loss (last 50): {avg_loss:.4f}")

            # Epoch 结束，主进程打印并保存检查点
            if num_batches == 0:
                if is_main_process:
                    logger.warning("本 epoch 未产生任何批次，可能数据量过小")
            else:
                avg_loss = total_loss / num_batches
                if is_main_process:
                    logger.info(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")

            # 使用 DeepSpeed 推荐的方式保存检查点（每个进程都可调用，内部会处理）
            save_tag = f"epoch{epoch+1}"
            engine.save_checkpoint("checkpoints", tag=save_tag)
            if is_main_process:
                logger.info(f"已保存检查点: checkpoints/{save_tag}")

        # 保存最终模型权重（仅主进程保存以免重复）
        if is_main_process:
            torch.save(engine.module.state_dict(), "models/final_model.pth")
            logger.info("最终模型已保存至 models/final_model.pth")

    finally:
        # 尝试优雅关闭 DeepSpeed / 分布式进程组，避免 PyTorch NCCL 的警告
        try:
            if engine is not None and hasattr(engine, 'save_checkpoint'):
                # DeepSpeed 有时会在 shutdown 时处理额外资源，尝试调用其 shutdown()（若存在）
                if hasattr(deepspeed, 'shutdown'):
                    try:
                        deepspeed.shutdown()
                        if is_main_process:
                            logger.info('已调用 deepspeed.shutdown()')
                    except Exception as e:
                        logger.warning(f'deepspeed.shutdown() 调用失败: {e}')
        except Exception:
            pass

        # PyTorch 分布式进程组销毁
        try:
            if dist.is_initialized():
                # 在销毁前执行 barrier 确保所有进程达到此处
                try:
                    dist.barrier()
                except Exception:
                    # barrier 在某些异常情况下可能失败，仍要尝试 destroy
                    pass
                try:
                    dist.destroy_process_group()
                    if is_main_process:
                        logger.info('已成功销毁 torch.distributed 进程组 (destroy_process_group)')
                except Exception as e:
                    logger.warning(f'调用 dist.destroy_process_group() 时出错: {e}')
        except Exception as e:
            logger.warning(f'检查或销毁分布式进程组时出错: {e}')


if __name__ == "__main__":
    main()
