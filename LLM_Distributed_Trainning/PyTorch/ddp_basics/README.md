# PyTorch DDP基础示例

PyTorch 的 `DistributedDataParallel`(DDP) 是一种基于**多进程**的分布式数据并行训练方式。它通过将数据划分为多个子批次，在每个 GPU 上独立计算前向和反向传播，然后使用高效的集合通信操作（如 `all_reduce`）同步梯度，确保所有模型副本的参数保持一致。

DDP 的训练流程涉及多个关键步骤，从环境初始化到最终的参数更新，下图概括了这一过程：

<img src="./imags/pytorch_01.png" alt="pytorch_01" style="zoom:40%;" />



## 代码功能分段说明

**分布式训练设置**：

- setup_distributed：初始化分布式环境，依赖 torchrun 提供的环境变量 RANK 和 WORLD_SIZE，使用 NCCL 后端。
- cleanup_distributed：清理分布式进程组。

**数据准备与 BPE 分词器**：

- prepare_data：加载 WikiText-2-raw-v1 或 WikiText-103-v1 数据集，提取文本。
- train_bpe_tokenizer：训练 BPE 分词器，保存为 JSON 文件。

**数据集处理**：

- TokenizedDataset：将文本通过 BPE 分词器编码，生成固定长度的训练块（block_size=128），支持因果语言建模。

**Transformer 模型定义**：

- CausalSelfAttention：实现因果自注意力机制。
- FeedForward：前馈网络层。
- TransformerBlock：Transformer 块，包含注意力层和前馈层。
- GPTLike：完整的 GPT-like 模型。

**训练函数**：

- main：主训练逻辑，解析命令行参数，设置 GPU 数量，加载数据和模型，进行多轮训练，支持单/多 GPU 模式和混合精度。



## 单机训练

**1. 安装依赖**：

```bash
pip install torch datasets tokenizers transformers
```

**2. 启动命令：**

默认使用所有 GPU（例如 2 个）。

```bash
torchrun --nproc_per_node=2 ddp_gpt_wikitext2.py
```

若要指定使用1个GPU，则使用如下命令。

```bash
torchrun --nproc_per_node=1 ddp_gpt_wikitext2.py
```

若要使用更多的自定义参数，可以参考如下命令。

```bash
torchrun --nproc_per_node=2 ddp_gpt_wikitext2.py --epochs 5 --batch_size 32 --vocab_size 5000
```



## 多机训练

**本示例中的主机环境：**

- 主机1：172.25.0.100，Ubuntu 2204，torch 2.8.0, cuda 12.8
- 主机2：172.25.0.200，Ubuntu 2204，torch 2.8.0, cuda 12.8



**注意事项：**

1. 多机训练时，必须要确保双方主机上的NCCL和CUDA版本一致，否则，极有可能导致通信或者训练异常。因此，在如下命令启动之前，请务必确保如下命令在各主机的结果相同：

   ```bash
   python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.version.nccl)"
   ```

2. 系统 CUDA 和 PyTorch 预编译 CUDA runtime 是两个独立的事物，前者主要用于让用户编译自己的 CUDA / C++ 代码，后者才是PyTorch 在运行时直接调用的库，它并不会依赖系统的 CUDA Toolkit。
   - 系统 CUDA（系统层面的 CUDA Toolkit）
     - 可通过 nvcc --version 命令查看和了解相关的版本信息
     - 包含的组件：CUDA 编译器、开发头文件、库文件（cuBLAS、cuDNN、NCCL 等的系统版）
     - PyTorch 运行时并不会强制使用系统CUDA，除非自行从源码编译 PyTorch
   - PyTorch 自带的 CUDA runtime（预编译包内置）
     - pip 或 conda 安装的 PyTorch 自带了一套固定版本的 CUDA runtime（cuBLAS、cuDNN、NCCL 等）
     - PyTorch 在运行时 直接调用这个自带的库，而不依赖系统的 CUDA Toolkit
     - 这意味着，用户只需要安装torch及其依赖的组件，无需关心系统上有没有安装 CUDA 或 NCCL，就能直接跑 GPU 程序



**主机1的启动命令：**

```bash
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=172.25.0.100 --master_port=29500 ddp_gpt_wikitext2.py --epochs 3 --batch_size 8
```

**主机2的启动命令：**

```bash
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr=172.25.0.100 --master_port=29500 ddp_gpt_wikitext2.py --epochs 3 --batch_size 8
```



**故障排查技巧：**

1. 若主机的网络环境是TCP/IP而非InfiniBand，需要在各主机上显式禁止InfiniBand，命令如下：

   ```bash
   export NCCL_IB_DISABLE=1  
   ```

2. 若参与训练的主机上不止一个网络接口，建议明确指定用于通信的接口名称，命令如下：

   ```bash
   export NCCL_SOCKET_IFNAME=eth0   # 将eth0替换为实际要使用的接口名称
   ```

3. 若需要打开调试日志以观测命令运行，可以使用如下命令进行（通常在rank0所在的节点运行即可）：

   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   export TORCH_DISTRIBUTED_DEBUG=DETAIL
   ```

4. 确认torchrun命令兼容当前主机环境中的rendezvous后端：

   ```bash
   python -c "import torch; import torch.distributed as dist; print(torch.__version__)"
   ```

5. 检查 MTU / offload 与网络性能问题（可选）

   ```bash
   # 查看 MTU
   ip -4 link show dev eth0
   
   # 关闭 TSO/GSO/GRO 临时测试
   sudo ethtool -K eth0 tso off gso off gro off
   # 重试训练，看是否影响
   # 恢复: sudo ethtool -K eth0 tso on gso on gro on
   ```

6. 确认各主机的conda/env与PyTorch/NCCL版本一致

   ```bash
   python - <<'PY'
   import torch, os
   print("torch", torch.__version__)
   print("nccl version:", torch.cuda.nccl.version() if hasattr(torch.cuda, 'nccl') else "n/a")
   print("cuda:", torch.version.cuda)
   print("gpu count", torch.cuda.device_count())
   PY
   ```

7. 





## 版权声明

本文档由[马哥教育](http://www.magedu.com)开发，允许自由转载，但必须保留马哥教育及相关的一切标识。另外，商用需要征得马哥教育的书面同意。
