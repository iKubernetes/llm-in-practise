# PyTorch Distributed分布式训练示例

PyTorch Distributed是PyTorch框架中的一个核心模块（torch.distributed 包），专为多进程并行计算设计，支持在单机或多机（多节点）环境下进行分布式训练和通信。它允许用户在多个计算节点上同步执行模型训练，主要通过提供通信原语和并行包装器（如 torch.nn.parallel.DistributedDataParallel，简称 DDP）来实现高效的分布式深度学习。该模块强调同步分布式训练，与其他并行方式（如 torch.multiprocessing 或 torch.nn.DataParallel）不同，它支持跨网络机器的协作，并要求为每个进程显式启动独立的训练脚本副本。

**关键组件：**

PyTorch Distributed 的核心在于进程管理和通信协调，以下是其主要组件：

- 初始化函数：
  - torch.distributed.init_process_group()：用于初始化默认分布式进程组，必须指定后端（backend）、初始化方法（init_method）、世界大小（world_size）和进程排名（rank）。该函数会阻塞直到所有进程加入，且非线程安全。支持的参数包括超时时间（timeout）和进程组选项（pg_options）。
  - torch.distributed.device_mesh.init_device_mesh()：初始化一个 DeviceMesh 对象，根据设备类型和网格形状（mesh shape）创建，支持 SPMD（Single Program Multiple Data）编程模型。所有进程的网格形状必须相同，否则可能导致挂起（hang）。它可以作为上下文管理器与 DTensor API 结合使用。
- 进程组（Process Groups）：
  - 默认进程组（世界组）：所有集体操作默认在此组上执行，要求所有进程参与。
  - 自定义进程组：通过 torch.distributed.new_group() 创建，允许在进程子集上进行细粒度通信，返回一个组句柄（group handle）用于集体调用。
- DeviceMesh：
  - 高层抽象，用于管理进程组或 NCCL 通信器，简化多维并行（如集群级）的设置。它尊重预选设备，并在分布式环境中自动处理资源分配。
- 分布式键值存储（Distributed Key-Value Store）：
  - 包括 TCPStore、FileStore 和 HashStore，用于进程间共享信息和初始化分布式包。支持操作如 set()、get()、add()、append() 等，部分操作（如 delete_key）仅限于特定存储类型。用于协调而非直接通信。
- Work 对象：
  - 表示异步操作的句柄，由非阻塞集体操作返回（如 all_reduce with async_op=True）。提供方法如 is_completed()（检查完成）、wait()（等待完成）、get_future() 和 get_future_result()（获取异步结果）。
- Reduce 操作（ReduceOp）：
  - 一个枚举类，支持 SUM、PRODUCT、MIN、MAX、BAND、BOR、BXOR 和 PREMUL_SUM 等操作。某些操作（如 AVG 和 PREMUL_SUM）仅限于 NCCL 后端和特定版本。

**用法：**

PyTorch Distributed 的典型用法涉及初始化、通信和清理。以下是步骤指南：

1. 初始化
   - 选择初始化方法：TCP（tcp://）、共享文件系统（file://）或环境变量（env://，默认）。环境变量需设置 MASTER_PORT、MASTER_ADDR、WORLD_SIZE 和 RANK。
   - 示例（TCP）：dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4)。
   - 初始化后，使用 torch.distributed.is_initialized() 检查状态，get_rank()、get_world_size() 和 get_backend() 获取进程信息。
2. 通信
   - 点对点通信：同步（send/recv）和异步（isend/irecv），支持张量交换。批处理操作如 batch_isend_irecv 处理多个操作。标签（tag）用于匹配，但 NCCL 不支持。
   - 集体通信：包括 broadcast（从源分发张量）、all_reduce（跨进程归约，如 SUM 操作）、reduce、all_gather、gather、scatter、reduce_scatter 和 all_to_all。示例：dist.all_reduce(tensor, op=dist.ReduceOp.SUM)。异步示例：work = dist.all_reduce(tensor, async_op=True)，然后 work.wait()。
   - 同步：使用 barrier() 同步所有进程；monitored_barrier() 添加超时调试，仅 Gloo 支持。
     对象通信：如 object_broadcast()，但限于可 pickle 对象。
3. 关闭和重新初始化
   - 使用 destroy_process_group() 清理资源，确保所有进程在超时内调用集体操作以避免挂起。
   - 重新初始化不支持或未测试，由于同步挑战，可能导致问题。

## 示例中的目录说明

- ddp_basics：DDP示例目录
- fsdp_basics：FSDP示例目录
- transformer_basics：GPT风格的Transformer架构模型的基础示例

