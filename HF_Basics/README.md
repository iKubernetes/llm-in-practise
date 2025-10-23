# HuggingFace训练示例



## 环境配置说明

需要注意，除了正常可用于训练的GPU外，有些工作站上很可能还存在一张“亮机卡”。若这张“亮机卡”过于老旧，可能会导致训练过程的启动异常。若存在该种多GPU异构（新卡+旧卡）的情况，则在分布式训练时，很可能会导致CUDA初始化失败的情况。下面我们通过一个示例来说明如何解决该问题。

例如，在某训练机上运行“nvidia-smi -L”命令，即结果显示如下：

```text
GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-ac50e9bd-5330-6ce2-2d4b-8f95df663b60)
GPU 1: NVIDIA GeForce GT 1030 (UUID: GPU-addf6be6-9a6e-c331-da9e-b0ab90813984)
GPU 2: NVIDIA GeForce RTX 3090 (UUID: GPU-ff67d0d7-648f-d302-c50e-49297407cd04)
```

这代表主机有三张GPU，其中两张为 RTX 3090（Compute Capability 8.6），还有一张为GT 1030（Compute Capability 6.1）。这种情况下，即便使用环境变量“export CUDA_VISIBLE_DEVICES=0,2”来声明只使用第0张和第2张GPU，GT 1030 (GPU 1) 就算未被选中，但CUDA初始化时仍可能被驱动探测、注册内核映像（尤其是某些版本的driver + CUDA runtime 会全局扫描 GPU）。

于是，结果就是CUDA 内核编译时默认针对最高可见架构（sm_86），但是在初始化或NCCL通信阶段，GT 1030无法执行sm_86的kernel，便会导致报错：“no kernel image is available for execution on the device”。

**解决方案一：物理或逻辑隔离**

如果 1030 是用于显示输出（常见于工作站）：

- 保留 GT 1030 连接显示器；

- 训练进程只用 3090；

- 显示环境可使用 Xorg 绑定 1030，训练环境只识别 3090。


下面是命令示例。其中，CUDA_DEVICE_ORDER决定了GPU的枚举顺序，而 CUDA_VISIBLE_DEVICES则是在这个新的顺序基础上指定哪些GPU对程序可见。

> 注意：如果不设置CUDA_DEVICE_ORDER，其默认值是FASTEST_FIRST，这时CUDA会按GPU的计算性能从高到低进行枚举，性能最强的GPU设备索引为0。这在GPU型号不同的混合系统中很常见。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2
torchrun --nproc_per_node=2 train.py
```

这样torchrun命令就会在两个3090上启动训练进程，忽略GT 1030。

**解决方案二：在 NCCL 层面屏蔽旧卡（若训练环境为多机分布式）**

即便有了前面解决方案一的设定，有时NCCL在初始化通信时仍会检测所有设备。此种情况下，可通过如下命令声明的环境变量来防止因NCCL探测GT 1030失败而报错。

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0  # 注意要修改为实际使用的网络接口的名称
```

- NCCL_P2P_DISABLE=1：这个设置会禁用NVIDIA GPU之间的点对点（P2P）直接通信。
  - 通常，支持NVLink的GPU或通过PCIe直接连接的GPU会尝试使用高速直连通道交换数据。禁用后，数据会通过主机内存中转，这可能更稳定，但速度通常更慢。
  - 在RTX 3090/4090/5090等消费级GPU的跨节点训练中，有时禁用P2P可以规避PCIe拓扑的带宽争抢问题。
- NCCL_IB_DISABLE=1：强制NCCL不使用InfiniBand（IB）网络进行跨节点通信，即使系统检测到可用的IB硬件。
  - 禁用后，NCCL会回退使用标准的IP网络（如以太网）
  - 这在没有InfiniBand支持的集群中是必须的，否则NCCL可能初始化失败或报错
- NCCL_SOCKET_IFNAME=eth0：这个设置用于明确指定NCCL进行跨节点通信时使用的网络接口。服务器有多个网络接口（如万兆以太网、InfiniBand的ib0接口）时，明确指定可以避免NCCL自动选择错误的接口，确保通信流量走在预期的物理网络上。

**多机分布式环境声明的环境变量（小结）：**

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
#export NCCL_BLOCKING_WAIT=1  # 即将废弃，并由TORCH_NCCL_BLOCKING_WAIT替代
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
```

- TORCH_NCCL_BLOCKING_WAIT=1：让PyTorch在发起NCCL操作后同步等待操作完成，以便于调试（否则NCCL操作是异步的）。常用于调试NCCL通信错误，确保错误信息能及时抛出并定位。
- NCCL_DEBUG=INFO：设置NCCL库的调试信息输出级别为INFO，会打印详细的通信建立、数据传输等信息。常用于排查NCCL通信问题，了解分布式训练中各节点的通信状态和性能。
- WANDB_DISABLED=1：完全禁用Weights & Biases的日志记录功能，防止其尝试初始化或上传数据。在不使用W&B平台进行实验跟踪时，该设置可避免其相关的登录错误或网络开销。
- TOKENIZERS_PARALLELISM=false：禁用HuggingFace Tokenizers库的并行处理（其默认会使用多线程进行tokenization），用以避免在多进程或特定环境中，tokenizer的多线程与外部多进程产生冲突，进而可能导致的死锁或错误。



## 启动训练

### 单机单卡训练

以前一节提到的多GPU主机环境为例，若要进行单机单GPU训练，可直接指定要使用的GPU，而后启动训练过程。例如下面的命令，表示在GPU 0启动训练。

```bash
export CUDA_VISIBLE_DEVICES=0
python trainer_evaluate_demo.py
```

Trainer会检测到单张 GPU，自动使用单卡模式训练（无任何DDP启动逻辑）。



### 单机多卡训练

#### 单机多卡训练（accelerate）

若要让两张RTX 3090 一起参与训练（例如前一节示例环境中的GPU 0 和 GPU 2），则不能再通过“python trainer_demo.py”命令进行，而是要改为使用accelerate launch命令进行。在使用accelerate config命令配置好训练环境后，即可使用如下命令启动分布式训练。

```bash
accelerate launch trainer_evaluate_demo.py
```

也可以使用命令选项指定使用多GPU训练，并指定要使用的进程数，例如“--multi_gpu --num_processes 2”。

#### 单机多卡训练(torchrun)

另外，多机多卡训练也可以使用PyTorch的分布式启动器torchrun命令进行。例如下面的命令，表示在GPU 0和GPU 2上进行分布式训练。

```bash
export CUDA_VISIBLE_DEVICES=0,2
torchrun --nproc_per_node=2 trainer_evaluate_demo.py
```

命令行选项“--nproc_per_node=2” 表示每张显卡启动一个进程，Trainer会自动检测并进入分布式模式。



### 多机多卡训练

使用accelerate launch命令的启动方法如下。

Master主机：

```bash
accelerate launch --main_process_ip "MASTER_IP" --main_process_port 29500 --num_processes 8 --num_machines 2 --machine_rank 0 trainer_evaluate_demo.py
```

其它主机（例如主机B）：

```bash
accelerate launch --main_process_ip "MASTER_IP" --main_process_port 29500 --num_processes 8 --num_machines 2 --machine_rank 1 trainer_evaluate_demo.py
```

