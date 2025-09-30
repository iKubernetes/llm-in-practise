# PyTorch FSDP实践示例

FSDP (Fully Sharded Data Parallelism，完全分片数据并行) 和 FSDP2 是 PyTorch 中用于训练超大模型的关键技术，它们通过将模型的参数 (Parameters)、梯度 (Gradients) 和 优化器状态 (Optimizer States) 分片（Sharding）到不同的 GPU 上，从而极大地减少了单个 GPU 的显存占用。

- FSDP 是 PyTorch 中分布式训练的一个主要功能，其核心思想是模仿 DeepSpeed ZeRO-3 的分片策略。

  - FlatParameter (扁平化参数)：这是FSDP1 的关键特性。它将一个 FSDP 模块（通常是模型的一个层或块）内所有的参数扁平化为一个巨大的 1D 连续张量 (FlatParameter)，然后对这个 1D 张量进行分片，分配给不同的 GPU。
  - 动态 All-Gather：
    - 在前向传播（Forward Pass）中，需要计算某一层时，每个 GPU 会通过 All-Gather 操作从其他 GPU 收集该层的全部参数（即 FlatParameter 的完整副本）。
    - 计算完成后，这些完整的参数副本会被释放，从而节省显存。
  - Reduce-Scatter：
    - 在反向传播（Backward Pass）中，每个 GPU 计算出完整的梯度后，使用 Reduce-Scatter 操作来聚合（Reduce）和分散（Scatter）梯度。每个 GPU 最终只保存其对应参数分片的梯度分片。
    - 这个过程确保了通信（All-Gather 和 Reduce-Scatter）可以与计算重叠，提高了训练效率。

- FSDP2 是 FSDP 的改进版本，旨在解决 FSDP1 的局限性，提供更简洁的 API 和更灵活的内部机制。

  - 核心优势：FSDP2 的主要突破在于引入了 DTensor (Distributed Tensor，分布式张量) 抽象层，并摒弃了 FlatParameter。

  

## FSDP（fsdp1）实践示例

### 单机训练

**1. 安装依赖**：

```bash
pip install torch datasets tokenizers transformers
```

**2. 启动命令：**

默认使用所有 GPU（例如 2 个）。

```bash
torchrun --nproc_per_node=2 fsdp_gpt_wikitext2.py
```

若要指定使用1个GPU，则使用如下命令。

```bash
torchrun --nproc_per_node=1 fsdp_gpt_wikitext2.py
```

若要使用更多的自定义参数，可以参考如下命令。

```bash
torchrun --nproc_per_node=2 fsdp_gpt_wikitext2.py --epochs 5 --batch_size 32 --vocab_size 5000
```



### 多机训练

**本示例中的主机环境：**

- 主机1：172.25.0.100，Ubuntu 2204，torch 2.8.0, cuda 12.8
- 主机2：172.25.0.200，Ubuntu 2204，torch 2.8.0, cuda 12.8



**注意事项：**

1. 多机训练时，必须要确保双方主机上的NCCL和CUDA版本一致，否则，极有可能导致通信或者训练异常。因此，在如下命令启动之前，请务必确保如下命令在各主机的结果相同：

   ```bash
   python -c "import torch; print(torch.__version__); print(torch.version.cuda);"
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
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=172.25.0.100 --master_port=29500 fsdp_gpt_wikitext2.py --epochs 3 --batch_size 8
```

**主机2的启动命令：**

```bash
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr=172.25.0.100 --master_port=29500 fsdp_gpt_wikitext2.py --epochs 3 --batch_size 8
```



## FSDP2实践示例

首先，验证fully_shard函数是否可用。

```bash
python -c "from torch.distributed.fsdp import fully_shard; print('fully_shard imported successfully')"
```

若是能够正常响应，则表示可以继续使用fsdp2。



### 单机训练

若要指定使用2个GPU，则使用如下命令。

```bash
torchrun --nproc_per_node=2 fsdp2_gpt_wikitext2.py
```

若要使用更多的自定义参数，可以参考如下命令。

```bash
torchrun --nproc_per_node=2 fsdp2_gpt_wikitext2.py --epochs 5 --batch_size 32 --vocab_size 5000
```



### 多机训练

**本示例中的主机环境：**

- 主机1：172.25.0.100，Ubuntu 2204，torch 2.8.0, cuda 12.8
- 主机2：172.25.0.200，Ubuntu 2204，torch 2.8.0, cuda 12.8



**主机1的启动命令：**

```bash
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=172.25.0.100 --master_port=29500 fsdp2_gpt_wikitext2.py --epochs 3 --batch_size 8
```

**主机2的启动命令：**

```bash
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr=172.25.0.100 --master_port=29500 fsdp2_gpt_wikitext2.py --epochs 3 --batch_size 8
```



## 故障排查技巧

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



## 版权声明

本项目及文档由[马哥教育](http://www.magedu.com)开发，允许自由转载，但必须保留马哥教育及相关的一切标识。另外，商用需要征得马哥教育的书面同意。
