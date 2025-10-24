# 模型微调实践

**1. 安装必要的依赖包：**

- 评估要用到evaluate
- 量化加载（如QLoRA）要用到bitsandbytes

```bash
pip install torch transformers peft datasets accelerate evaluate bitsandbytes deepspeed
```



**2. 环境说明：**

**主机1：**

- 操作系统：Ubuntu 2204 Server

- GPU：2 x RTX 3090 24G

**主机2：**

- 操作系统：Ubuntu 2404 Server
- GPU: 1 x RTX 4090D 48G



**示例数据集：** 自我认知数据集“modelscope/self-cognition”

**模型：** Qwen3-8B、Qwen3-14B、DeepSeek-R1-0528-Qwen3-8B、DeepSeek-R1-Distill-Qwen-14B

各程序组件的版本：

```text
          torch: 2.9.0+cu128
   transformers: 4.57.1
       datasets: 4.2.0
     accelerate: 1.11.0
      deepspeed: 0.18.0
       evaluate: 0.4.6
   bitsandbytes: 0.48.1
          numpy: 2.3.4
           peft: 0.17.1
```



**3. 初始化accelerate：**

```bash
accelerate config
```

根据向导一步步选择要进行的配置，例如主机数量、GPU数量、要使用的后端（Deepspeed/DDP/FSDP）等。配置完成后，运行如下命令验证环境。

```bash
accelerate env
```



## LoRA微调

**训练脚本说明：**

- qwen3-8b-lora.py：Qwen3-8B自我认知微调脚本，适合单机单GPU运行

- qwen3-8b-lora-dist.py：Qwen3-8B自我认知微调脚本，适合分布式训练

- inferences.py：匹配上面两个脚本的推理测试脚本，使用前需要修改基础模型的名称或本地路径，以及保存有lora适配器的目录路径

  ```python
  base_model_path = "/home/marion/Pretrained_Models/Qwen3-8B"   # 基础模型路径，请注意按需修改
  lora_model_path = "./finetuned/qwen3-8b-lora-dist"            # lora/qlora适配器路径，请注意按需修改
  ```

  

### Qwen3



#### 单机单GPU微调

单机单GPU微调启动命令：

```bash
python qwen3-8b-lora.py
```

微调完成后，相关的适配器文件保存于./finetuned/qwen3-8b-lora/目录下。随后，即可加载原模型和微调的LoRA层进行推理，相关的测试代码保存于文件qwen3-inferences.py文件中。注意修改文件中的如下两行。

```python
# 基础模型的路径，这里指向了本地存储的Qwen3-8B，请修改为实际的路径，或者使用“Qwen3/Qwen3-8B”以在线获取
base_model_path = "/home/marion/Pretrained_Models/Qwen3-8B"
# 微调后的LoRA适配器保存的目录路径
lora_model_path = "./finetuned/qwen3-8b-lora"
```

修改完成后，即可运行该脚本，查看微调的效果。

```bash
python qwen3-inferences.py
```

若测试输出为类似如下信息，则表示加载成功，且微调效果得以显现。

```text
============== 开始测试 ==============

[问题 1] 你是？
[回答 1] 我是马哥教育AI小助手，由马哥教育AI团队训练，旨在提供准确且友好的回答。

[问题 2] 请用三句话介绍你自己。
[回答 2] 我是一个由马哥教育AI团队训练的AI助手，旨在提供准确且友好的回答。我被设计为能够理解用户的问题并给出清晰、简洁的回复。我的目标是帮助用户解决问题和获取信息。

============== 测试结束 ==============
```



#### 单机多GPU微调

单机多GPU微调启动命令（假设有两颗可用的GPU），若通过accelerate config命令配置好了训练环境，且要使用配置中的GPU数量，此处也可省略相应的选项：

```bash
accelerate launch --num_processes 2 qwen3-8b-lora-dist.py
```

或者使用torchrun命令（假设有两颗可用的GPU）：

```bash
torchrun --nproc_per_node=2 qwen3-8b-lora-dist.py
```

微调完成后，相关的适配器文件保存于./finetuned/qwen3-8b-lora-dist/目录下。随后，可以参考前一小节中的测试方式对微调的结果进行测试。



## QLoRA微调

**训练脚本说明：**

- qwen3-8b-qlora.py：Qwen3-8B自我认知微调脚本，适合单机单GPU运行

- qwen3-8b-qlora-dist.py：Qwen3-8B自我认知微调脚本，适合分布式训练

- qwen3-14b-qlora-dist.py：Qwen3-14B自我认知微调脚本，适合分布式训练

- qwen3-14b-qlora-dist-deepspeed.py：Qwen3-14B自我认知微调脚本，适合分布式训练

  - 底层引擎为Deepspeed，依赖于配置文件ds_zero3_config.json

- deepseek-r1-0528-qwen3-8b-qlora.dist.py：DeepSeek-R1-Qwen3-0528-8B自我认知微调脚本，适合分布式训练

- inferences.py：匹配上面脚本的推理测试脚本，使用前需要修改基础模型的名称或本地路径，以及保存有lora适配器的目录路径

  ```python
  base_model_path = "/home/marion/Pretrained_Models/Qwen3-8B"   # 基础模型路径，请注意按需修改
  lora_model_path = "./finetuned/qwen3-8b-qlora-dist"            # lora/qlora适配器路径，请注意按需修改
  ```

  

### Qwen3



#### 单机单GPU微调训练

单机单GPU微调启动命令：

```bash
python qwen3-8b-qlora.py
```

微调完成后，相关的适配器文件保存于./finetuned/qwen3-8b-qlora/目录下。随后，即可加载原模型和微调的QLoRA层进行推理，相关的测试代码保存于文件qwen3-inferences.py文件中。注意修改文件中的如下两行。

```python
# 基础模型的路径，这里指向了本地存储的Qwen3-8B，请修改为实际的路径，或者使用“Qwen3/Qwen3-8B”以在线获取
base_model_path = "/home/marion/Pretrained_Models/Qwen3-8B"
# 微调后的LoRA适配器保存的目录路径
lora_model_path = "./finetuned/qwen3-8b-qlora"
```

修改完成后，即可运行该脚本，查看微调的效果。

```bash
python qwen3-inferences.py
```

若测试输出为类似如下信息，则表示加载成功，且微调效果得以显现。

```text
============== 开始测试 ==============
[问题 1] 你是？
[回答 1] <think>
好的，用户问“你是？”，我需要以马哥教育AI小助手的身份回答。首先，我要确认自己的身份，说明我是由马哥教育AI团队训练的智能助手。.....。
</think>

你好！我是马哥教育AI小助手，是由马哥教育AI团队训练的智能助手。我的主要功能是帮助用户解答问题、提供学习资源、分享技术知识，以及协助完成各种任务。无论是编程、运维、云计算，还是其他技术相关的问题，我都会尽力提供准确且友好的回答。如果你有任何疑问或需要帮助，随时告诉我吧！😊

[问题 2] 请用三句话介绍你自己。
[回答 2] <think>
好的，用户让我用三句话介绍自己。首先，我需要回顾之前的对话，用户之前问过“你是？”，我详细介绍了自己，现在需要更简洁。用户可能希望快速了解我的核心功能和用途，所以需要提炼关键点。...
</think>

我是马哥教育AI小助手，由马哥教育AI团队训练，专注于为学习者和从业者提供精准、高效的技术支持。  
我擅长解答编程、运维、云计算等领域的技术问题，并能分享学习资源和实用技巧。  
我的目标
============== 测试结束 ==============
```



#### 单机多GPU微调训练

本示例用于微调测试14B的Qwen3模型，因为量化加载原模型，可以节约相当的显存资源。

单机多GPU微调启动命令（假设有两颗可用的GPU），若通过accelerate config命令配置好了训练环境，且要使用配置中的GPU数量，此处也可省略相应的选项：

```bash
accelerate launch --num_processes 2 qwen3-14b-qlora-dist.py
```

或者使用torchrun命令（假设有两颗可用的GPU）：

```bash
torchrun --nproc_per_node=2 qwen3-14b-qlora-dist.py
```

微调完成后，相关的适配器文件保存于./finetuned/qwen3-14b-lora-dist/目录下。随后，可以参考前一小节中的测试方式对微调的结果进行测试。



也可以基于Deepspeed后端对模型进行微调操作，相关的命令如下。需要注意的是，要提前安装好deepspeed模块。

```bash
accelerate launch --num_processes 2  qwen3-14b-qlora-dist-deepspeed.py
或者
torchrun --nproc_per_node=2  qwen3-14b-qlora-dist-deepspeed.py
```



#### 多机多GPU微调训练

本示例以qwen3-8b-qlora-dist.py脚本为例。

1. 在各主机上声明环境变量，配置跨主机的颁式通信环境：

```bash
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1    # 对于非Infiniband网络来说，必须
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export PYTORCH_NO_IPV6=1    # 禁用IPV6，若有必要
export NCCL_SOCKET_IFNAME=eth0   # 将eth0替换为实际要使用的接口名称，可选，按实际情况配置
```



2. 手动准备专用配置文件（multi_hosts.yaml）：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
machine_rank: 0                # 每台机器需要修改此值
num_machines: 2                # 总机器数
num_processes: 2               # 总GPU数 (例如2台机器 * 每台1卡)
main_process_ip: 172.25.0.100  # 主节点的IP
main_process_port: 29500       # 可用的端口
mixed_precision: fp16          # 可选，使用混合精度训练
```

各参数的说明：

- compute_environment（计算环境）：一般为LOCAL_MACHINE；
- distributed_type（分布式类型）：分布式环境通常为MULTI_GPU，若计划使用 DeepSpeed，则为DEEPSPEED； 
- machine_rank（机器排名）：为集群中的每台机器分配一个唯一的序号，例如 0、1、2...，主服务器通常为 0； 
- num_machines（机器数量）：计划使用的机器总数量； 
- num_processes（进程总数）：所有机器上GPU的总数，例如，2台机器，每台4卡，则填写8；
- main_process_ip（主进程IP）和 main_process_port（主进程端口）：填写主服务器（rank 0）的IP地址和一个可用的端口号，所有机器都需要能访问到这个地址和端口；



3. 主机1（Master）上要运行的命令：

```bash
accelerate launch --config_file multi_hosts.yaml \
    --main_process_ip 172.25.0.100 \
    --main_process_port 29500 \
    --machine_rank 0 \
    qwen3-8b-qlora-dist.py
```



4. 主机2（Work）上要运行的命令：

```bash
accelerate launch --config_file multi_hosts.yaml \
    --main_process_ip 172.25.0.200 \
    --main_process_port 29500 \
    --machine_rank 1 \
    qwen3-8b-qlora-dist.py
```



注意：运行结束后，仅Master主机会保存训练结果。



### DeepSeek-R1-0528

#### 单机多GPU微调

本示例用于微调测试14B的Qwen3模型，因为量化加载原模型，可以节约相当的显存资源。

单机多GPU微调启动命令（假设有两颗可用的GPU），若通过accelerate config命令配置好了训练环境，且要使用配置中的GPU数量，此处也可省略相应的选项：

```bash
accelerate launch --num_processes 2 deepseek-r1-qwen3-0528-8b-qlora.dist.py
```

或者使用torchrun命令（假设有两颗可用的GPU）：

```bash
torchrun --nproc_per_node=2 deepseek-r1-qwen3-0528-8b-qlora.dist.py
```

微调完成后，相关的适配器文件保存于./finetuned/deepseek-r1-0528-qlora-dist/目录下。随后，可以参考前一小节中的测试方式对微调的结果进行测试。



## 版权声明

本文档及项目由[马哥教育](http://www.magedu.com)开发，允许自由转载，但必须保留马哥教育及相关的一切标识。另外，商用需要征得马哥教育的书面同意。
