# DeepSpeed实践案例

DeepSpeed 是一款由 Microsoft 开发的开源深度学习优化软件套件，旨在为大规模深度学习模型的训练和推理提供极致的速度和规模支持。它特别适用于训练和推理参数规模达数十亿甚至万亿的稠密或稀疏模型，能够高效扩展到数千个 GPU，并在资源受限的 GPU 系统上实现高性能训练和推理。



## DeepSpeed简介

**分布式训练的关键特性：**

DeepSpeed 在分布式训练方面提供了多项先进功能，使其成为训练大型模型的首选工具：

- 支持超大规模模型：能够处理参数量达万亿级的模型，支持在数千个 GPU 上高效扩展，确保训练过程的高吞吐量和低延迟。
- ZeRO（Zero Redundancy Optimizer）：一种内存优化技术，通过分区优化器状态、梯度和参数，减少内存冗余，支持在有限内存中训练更大模型。ZeRO-Infinity 进一步扩展了这一功能，允许利用 CPU 和 NVMe 存储来处理更大规模。
- 3D-Parallelism：结合数据并行、管道并行和张量并行，实现高效的分布式训练，优化通信开销并提升整体性能。
- DeepSpeed-MoE（Mixture of Experts）：专为稀疏模型设计，支持专家并行，显著提高训练效率，尤其适用于大型语言模型。
- 并行技术集成：包括张量并行、管道并行、专家并行和 ZeRO 并行，结合自定义推理内核和通信优化，实现低延迟和高吞吐量的推理。
- 压缩技术：如 ZeroQuant 和 XTC（Extreme Tensor Compression），用于加速推理、缩小模型大小并降低压缩成本，支持在资源受限的环境中部署。
  这些特性使 DeepSpeed 在分布式环境中能够实现高效的模型分区、通信最小化和资源利用最大化，适用于从单个 GPU 到大规模集群的各种场景。

**核心组件：**

DeepSpeed 的架构围绕四个创新支柱构建，每个支柱专注于特定方面：

1. DeepSpeed-Training：专注于大规模训练的系统创新，包括 ZeRO、3D-Parallelism、DeepSpeed-MoE 和 ZeRO-Infinity。这些组件帮助用户高效训练大型模型，详细文档可在官网的 Training 部分找到。
2. DeepSpeed-Inference：针对推理优化，结合高性能内核和异构内存技术，实现低延迟和高吞吐量，支持数千模型的快速部署。
3. DeepSpeed-Compression：提供易用的压缩工具，如 ZeroQuant 和 XTC，用于减少模型大小和推理延迟，同时保持高准确性。
4. DeepSpeed4Science：一个针对科学领域的倡议，帮助领域专家构建 AI 系统解决科学挑战，包括专属教程和资源。
   此外，DeepSpeed 软件套件包括：
   - DeepSpeed Library：开源仓库，实现训练、推理和压缩的创新功能，支持轻松组合特性（GitHub：https://github.com/microsoft/DeepSpeed）。
   - Model Implementations for Inference (MII)：用于低延迟、高吞吐量推理的开源仓库，支持数千模型的最小代码优化。

**支持的框架与集成**

DeepSpeed 与多个流行开源深度学习框架无缝集成，便于用户在现有工作流中使用：

- Hugging Face Transformers：通过 DeepSpeed 集成，支持高效训练和推理。
- Hugging Face Accelerate：提供 DeepSpeed 的使用指南。
- PyTorch Lightning：通过 DeepSpeedStrategy 策略集成。
- MosaicML：在 Trainer 中支持 DeepSpeed 集成。

这些集成允许用户以最小修改即可利用 DeepSpeed 的功能。此外，DeepSpeed 推荐在 Azure 上使用，通过 AzureML 配方提供作业提交和数据准备脚本。



## 本示例说明

本示例基于一个常规的Decoder-Only的Transformer模型进行DeepSpeed改造，使其分别能够支持ZeRO Stage-1/2/3和Offload。

- GPTLike-Base：示例中使用的基础Tranformer Decoder-Only模型；
- DeepSpeed-GPTLike-ZeRO-1：改造GPTLike-Base中的模型，支持能够基于DeepSpeed的ZeRO Stage-1策略进行分布式训练； 
- DeepSpeed-GPTLike-ZeRO-2 ：基于DeepSpeed的ZeRO Stage-2策略对GPTLike-Base中定义的模型进行分布式训练； 
- DeepSpeed-GPTLike-ZeRO-3 ：基于DeepSpeed的ZeRO Stage-3策略对GPTLike-Base中定义的模型进行分布式训练； 
- DeepSpeed-GPTLike-ZeRO-Offload ：基于DeepSpeed的ZeRO Offload策略对GPTLike-Base中定义的模型进行分布式训练； 
- DeepSpeed-GPTLike-Multihosts：基于ZeRO-2的多主机分布式训练示例



### 准备DeepSpeed环境

首先，创建专用于DeepSpeed的虚拟Python环境。

```bash
conda create -n deepspeed python=3.11
conda activate deepspeed
```

而后，在deepspeed虚拟环境中安装相关的Python模块。

```bash
pip install torch deepspeed datasets transformers
```

### deepspeed命令

DeepSpeed 命令的基本结构如下：

```bash
deepspeed [LAUNCHER_OPTIONS] <training_script.py> [SCRIPT_ARGS]
```

- [LAUNCHER_OPTIONS]：DeepSpeed 启动器的选项，用于配置分布式环境、节点数量、GPU 分配等。这些选项以 -- 开头，用于控制 DeepSpeed 的运行行为。
- <training_script.py>：训练脚本路径（例如 train.py）。脚本需适配 DeepSpeed API（如使用 deepspeed.initialize() 初始化模型和优化器）。
- [SCRIPT_ARGS]：传递给训练脚本的自定义参数（例如 --model_name bert --batch_size 32）。这些参数会通过 argparse 在脚本中解析。

DeepSpeed 命令的常用选项：

- --num_nodes=\<value>：指定参与训练的节点（机器）数量。等效于 torchrun 的 --nnodes，但不支持范围（弹性模式需使用 --include 或 --exclude）。
- --num_gpus=\<value>：每个节点使用的GPU数量，等效于torchrun的--nproc_per_node。如果未指定，DeepSpeed会尝试使用所有可用GPU。
- --hostfile=\<path>：指定主机文件路径，文件中列出参与训练的节点 IP/主机名及其 GPU 分配（slots）。
- --master_addr=\<host>：主节点的 IP 地址或主机名，用于分布式通信（等效于 torchrun 的 --master_addr），默认值为127.0.0.1。
- --master_port=\<port>：主节点的通信端口，用于分布式通信（等效于 torchrun 的 --master_port）。
- --deepspeed：显式启用 DeepSpeed 模式（通常自动检测），若脚本已包含 deepspeed.initialize()，此选项可省略。
- --log_dir=\<path>：日志文件存储目录，每个进程生成单独日志文件。
- --verbose：启用详细日志输出，显示 DeepSpeed 内部状态。
- --launcher=\<value>：指定启动器类型，支持 local（默认）、slurm、或 pdsh（Parallel Distributed Shell），默认为local。

### torchrun命令

DeepSpeed 命令的基本结构如下：

```bash
torchrun [OPTIONS] <training_script.py> [SCRIPT_ARGS]
```

- [OPTIONS]：torchrun 的命令行选项，用于配置分布式训练环境（如节点数、进程数、通信参数等）。
- <training_script.py>：要运行的Python脚本（例如train.py），必须是支持PyTorch分布式训练的脚本（通常包含 torch.distributed.init_process_group() 或类似逻辑）。
- [SCRIPT_ARGS]：传递给 \<script> 的参数，通常是脚本自定义的命令行参数（如 --epochs 10）。

torchrun命令的常用选项：

- --nnodes=\<value>：指定参与分布式训练的节点（机器）数量。可以是单一值（固定节点数）或范围（弹性模式，例如 2:4 表示最小 2 个节点，最大 4 个节点）。
- --nproc_per_node=\<value>：每个节点上启动的进程数。通常对应 GPU 数量（每个进程绑定一个 GPU），但也可以用于 CPU 训练。
- --node_rank=\<value>：当前节点的排名（从 0 开始）。多节点训练时，每个节点必须指定唯一的 node_rank。
- --master_addr=\<host>：主节点的 IP 地址或主机名，用于分布式通信（通常与 rdzv_backend=static 配合）。默认值：127.0.0.1（单节点默认 localhost）。
- --master_port=\<port>：主节点的通信端口，用于分布式通信。
- --start_method=\<value>：多进程启动方式，支持 spawn 或 fork（Python 多进程模块的启动方式），默认为spawn。
- --log_dir=\<path>：日志文件存储目录，每个进程会生成单独的日志文件。
- --local_addr=\<value>：指定本地节点的网络接口地址（用于多网络接口的机器）。



### 单主机多GPU训练

使用deepspeed命令或torchrun命令都可以，例如，在拥有着三个GPU的主机172.25.0.100上运行如下命令，可以使用0和2号GPU启动单主机多GPU训练：

```bash
export CUDA_VISIBLE_DEVICES=0,2 NCCL_DEBUG=INFO
deepspeed --num_gpus 2 DeepSpeed-GPTLike-ZeRO-1.py
```

如有必要，还可以传递脚本选项来测试对不同规格的模型进行训练。



### 多主机训练

#### 示例环境说明

##### 主机

| 主机名称 | 地址         | 操作系统    | GPU数量 | GPU型号        | CUDA版本 | GPU驱动版本 | Pytorch版本 | DeepSpeed版本 |
| -------- | ------------ | ----------- | ------- | -------------- | -------- | ----------- | ----------- | ------------- |
| 主机1    | 172.25.0.100 | Ubuntu 2204 | 3       | RTX 3090 24G   | 12.8     | 570.124.06  | 2.8.0       | 0.17.6        |
| 主机2    | 172.25.0.200 | Ubuntu 2204 | 1       | RTX 4090 D 48G | 12.8     | 570.124.06  | 2.8.0       | 0.17.6        |

##### 环境配置

1. 科学上网环境，以便能从HuggingFace下载需要的组件，例如Wikitext2数据值；
2. 主机上关闭防火墙，或者开放要使用的通信端口，例如TCP/29500；
3. 若主节点（MASTER_ADDR指定的主机）存在多个网络接口， 需要使用环境变量NCCL_SOCKET_IFNAME明确予以指定；
4. 若使用的TCP/IP网络环境，需要定义环境变量NCCL_IB_DISABLE的值为1来禁用InfiniBand网络；
5. 若需要观测训练过程中的详细日志信息，可使用环境变量NCCL_DEBUG进行指定，支持的日志级别包括：
   - INFO：输出初始化、通信等关键信息。
   - TRACE：输出详细调试信息（如每个通信步骤）。
   - WARN、ERROR：仅输出警告或错误。
6. 若要明确指定节点上使用的GPU设备，可以使用环境变量CUDA_VISIBLE_DEVICES来定义，例如“export CUDA_VISIBLE_DEVICES=0,2”表示只有0和2号GPU对命令可见

#### 启动命令

主机1（rank0）：

```bash
#export NO_PROXY="localhost,127.0.0.0/8,::1,172.25.0.0/16,192.168.0.0/16,10.0.0.0/8" HTTPS_PROXY=http://127.0.0.1:8889/ HTTP_PROXY=http://127.0.0.1:8889/
export NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,2 NCCL_SOCKET_IFNAME=eno1
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=172.25.0.100 --master_port=29500 DeepSpeed-GPTLike-ZeRO-2.py --ds_config ds_config.json --n_layer 6 --d_model 768 --n_head 12
```

主机2（rank1）：

```bash
#export NO_PROXY="localhost,127.0.0.0/8,::1,172.25.0.0/16,192.168.0.0/16,10.0.0.0/8" HTTPS_PROXY=http://127.0.0.1:8889/ HTTP_PROXY=http://127.0.0.1:8889/
export NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr=172.25.0.100 --master_port=29500 DeepSpeed-GPTLike-ZeRO-2.py --ds_config ds_config.json --n_layer 6 --d_model 768 --n_head 12
```





### 一些测试工具



```bash
python -c 'import deepspeed; print(deepspeed.__version__)'"
```

```bash
ssh 172.25.0.200 "source /home/marion/miniconda3/bin/activate deepspeed && python -c 'import deepspeed; print(deepspeed.__version__)'"
```



```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```





环境验证脚本

```bash
#!/bin/bash
# check_env.sh
echo "===== Host: $(hostname) ====="
echo "GPU Count: $(nvidia-smi -L | wc -l)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
```



测试另一主机的商品连通性。

```bash
nc -zv 172.25.0.100 29500
```

