# Transformer模型示例

Decoder-Only风格的Tranformer模型训练代码示例。

## GPTLike模型

在模型的定义中使用了标准的MHA和FFN。

### 模型介绍

#### 整体架构（GPTLike 类）

**输入嵌入**（Token Embedding 和 Position Embedding）：

- tok_emb = nn.Embedding(vocab_size, d_model)：将输入 token ID 映射到高维嵌入向量（维度为 d_model，默认为 512）。这允许模型处理离散的词汇表（vocab_size 默认为 3000）。

- pos_emb = nn.Embedding(block_size, d_model)：添加位置嵌入，用于捕捉序列中 token 的相对位置信息。因为 Transformer 本身不具备位置敏感性，这是一个关键特征，确保模型区分不同位置的相同 token。
- 权重共享：输出头 (head) 的权重与 token 嵌入权重共享（self.head.weight = self.tok_emb.weight）。这是一个优化技巧，减少参数量并提升泛化能力，常见于 GPT 系列模型中。

**Dropout** 层：

- drop = nn.Dropout(dropout)：在嵌入层后应用 dropout（默认为 0.1），防止过拟合。通过随机丢弃部分神经元，提升模型鲁棒性。


**Transformer** 块序列：

- blocks = nn.ModuleList([TransformerBlock(...) for _ in range(n_layer)])：由多个 TransformerBlock 堆叠而成（n_layer 默认为 6）。每个块处理输入序列，实现自注意力计算和非线性变换。这是模型的核心，允许逐层提取更高级的特征表示。


**最终层归一化和输出头**：

- ln_f = nn.LayerNorm(d_model)：在所有 Transformer 块后应用层归一化，稳定梯度流动。

- head = nn.Linear(d_model, vocab_size, bias=False)：线性层将隐藏状态投影回词汇表维度，生成 logits（用于 softmax 预测下一个 token）。无偏置设计简化模型，并与权重共享相匹配。

**参数配置灵活性**：

- 支持自定义超参数：vocab_size（词汇大小）、block_size（最大序列长度，默认为 128）、n_layer（层数）、n_head（注意力头数，默认为 8）、d_model（隐藏维度）、dropout（丢弃率）。

- 总参数量大致为：嵌入层 + 位置嵌入 + (每个 Transformer 块的参数) × n_layer + 输出头。典型配置下参数量在数百万级别，适合单机训练。



#### TransformerBlock 模块

**层归一化**（LayerNorm）：

- ln1 和 ln2：分别在自注意力和前馈网络前应用 LayerNorm（nn.LayerNorm(d_model)）。这有助于缓解梯度消失/爆炸问题，并加速训练收敛。LayerNorm 在每个 token 的特征维度上独立归一化。


**自注意力子模块**（CausalSelfAttention）：

- 这是 Transformer 的核心机制，使用 nn.MultiheadAttention 实现（embed_dim = d_model, num_heads = n_head）。

- 因果掩码（Causal Masking）：在 forward 中生成上三角掩码（torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()），确保注意力只关注当前和过去的 token，防止未来信息泄露。这使得模型适合自回归生成（如 GPT 的 “从左到右” 预测）。
- 多头注意力：将查询（Query）、键（Key）和值（Value）拆分成 n_head 个头（默认为 8），并行计算注意力分数，提升模型捕捉多方面关系的容量。
- Dropout：注意力 dropout (attn_dropout=0.0) 和残差 dropout (resid_dropout=dropout=0.1)，进一步正则化。

**前馈网络子模块**（FeedForward）：

- 由两层线性层组成：nn.Linear(d_model, hidden_dim) 和 nn.Linear(hidden_dim, d_model)，中间使用 GELU 激活（nn.GELU()）。

- 隐藏维度扩展：hidden_dim = int(d_model * mlp_ratio)（mlp_ratio 默认为 4.0），即隐藏层维度为 d_model 的 4 倍（默认为 2048），增加非线性容量。
- Dropout：在输出后应用 dropout，防止过拟合。

**残差连接**（Residual Connections）：

- 在注意力后：x = x + self.attn(self.ln1(x))。

- 在前馈后：x = x + self.mlp(self.ln2(x))。
- 这允许梯度直接流动到浅层，促进深层网络的训练稳定性，是 Transformer 成功的关键特征之一。



#### 权重初始化（_init_weights 方法）

**自定义初始化**：通过 self.apply(self._init_weights) 应用于所有子模块。

- 线性层和嵌入层：使用正态分布初始化权重（mean=0.0, std=0.02），这是一个小标准差策略，防止初始梯度过大。

- 线性层偏置：初始化为零（nn.init.zeros_(module.bias)）。
- LayerNorm：偏置初始化为零，权重初始化为 1（nn.init.ones_(module.weight)），保持初始分布不变。

这个初始化方案受 GPT 论文启发，确保模型从一个合理的起点开始训练，避免随机性导致的收敛问题。



#### 前向传播（forward 方法）

**输入处理**：

- 输入 idx：批次大小 B × 序列长度 L 的 token ID 张量。

- 如果 L > block_size，截断序列（idx = idx[:, :self.block_size]），防止超出位置嵌入范围。

**嵌入计算**：

- token 嵌入 + 位置嵌入：x = self.tok_emb(idx) + self.pos_emb(pos)，其中 pos 是序列索引张量（0 到 L-1）。

- 应用 dropout：x = self.drop(x)。

**逐层处理**：

- 通过所有 TransformerBlock：for blk in self.blocks: x = blk(x)，逐层精炼表示。


**输出生成**：

- 最终 LayerNorm：x = self.ln_f(x)。

- 投影到 logits：logits = self.head(x)，形状 B × L × vocab_size，用于计算交叉熵损失（在脚本中与目标 y 比较）。

**效率考虑**：使用 non_blocking=True 在数据转移到设备时，避免阻塞；序列长度检查确保安全性。



#### 与其他 GPT 模型的比较与优势

**相似性**：类似于 OpenAI 的 GPT-2/3，采用 decoder-only Transformer、因果注意力、位置嵌入和权重共享。
**简化点**：无额外的偏置、较小的默认规模（6 层，512 维），适合教育和实验目的。未包括高级特征如 rotary 位置嵌入或 flash attention。
**优势**：

- 模块化：易于扩展（如增加层数或头数）。

- 高效训练：残差和 LayerNorm 确保稳定；权重共享减少参数（约节省 vocab_size * d_model 参数）。
- 适用性：结合 BPE 分词器，适合处理自然语言序列；block_size 限制序列长度，控制内存使用。

**潜在局限**：固定位置嵌入限制了泛化到更长序列；无预训练权重，需从头训练。



#### 在训练脚本中的集成

- 模型在 main 函数中实例化，并移动到设备（CUDA 或 CPU）。
- 损失计算：使用 nn.CrossEntropyLoss()，针对 logits 和目标 token（y 是 x 的偏移版本）。
- 优化：AdamW 优化器，带权重衰减；StepLR 调度器逐步降低学习率（gamma=0.95）。
- 梯度裁剪：nn.utils.clip_grad_norm_ 防止梯度爆炸。
- 检查点保存：包括模型状态、优化器和调度器，便于恢复训练。

### 模型训练脚本

相关的模型训练代码文件有如下几个：

- GPTLike_wikitext2.py：极简的模型定义
- GPTLike_wikitext2_fixed_pe.py：适合训练使用的模型脚本，使用了固定正余弦位置编码
- GPTLike_wikitext2_learned_pe.py：适合训练使用的模型脚本，使用了可学习的位置编码

后两个模型训练代码支持使用命令行选项对模型规格等进行配置，这些参数共同控制模型的训练行为、架构规模和数据处理方式：

- 训练相关（epochs, batch_size, lr, weight_decay, clip_grad_norm）：影响优化过程和收敛速度。
- 模型架构（n_layer, n_head, d_model, dropout, block_size, vocab_size）：决定模型容量和表达能力。
- 其他（seed, save_interval, save_dir）：确保可重复性与训练状态管理。

#### 启动训练的命令

1. 基于默认配置启动：

   ```bash
   python SCRIPT_NAME.py
   ```

2. 以指定配置参数的方式启动的示例（仅后两个脚本支持）：

   ```bash
   python SCRIPT_NAME.py --epochs 10 --batch_size 16 --block_size 512 --lr 1e-3 --n_layer 12 --d_model 1024 --dropout 0.05
   ```

   

## DeepSeekLike模型

在模型层的定义中，使用MoE替代了FFN，使用MLA替代了标准的MHA。

