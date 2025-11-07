# 模型量化示例



## 基于GPTQModel的GPTQ量化示例



#### 安装gptqmodel模块

首先，我们使用以下pip命令来安装 GPTQModel，其中的--no-build-isolation选项在某些情况下有助于避免构建隔离问题，而 -v参数会让安装过程输出更详细的信息，方便排查可能出现的错误。

```bash
# 若需要，可以指定gptqmodel的版本，本示例使用的版本是4.2.5
pip install gptqmodel --no-build-isolation -v
# 或，使用如下命令安装GPTQModel及所有依赖
pip install gptqmodel[all] --no-build-isolation -v
```

如果需要从源码安装（例如为了获取最新的开发版功能或进行定制化修改），可以执行以下命令：

```bash
git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel

# python3-dev is required, ninja is to speed up compile
apt install python3-dev ninja

# pip: compile and install
# You can install optional modules like  vllm, sglang, bitblas.
# Example: pip install -v --no-build-isolation .[vllm,sglang,bitblas]
pip install -v . --no-build-isolation
```

#### 执行量化

示例中准备了脚本quantize_qwen3_8b_gptq.py，用于对模型Qwen3/Qwen3-8B模型进行4bit量化。

```bash
python quantize_qwen3_4b_gptq.py
```

量化后，可以加载量化模型进行推理测试。下面的命令通过推理脚本加载量化模型并进行测试。

```bash
python inference_qwen3_4b_gptq.py
```



## LLM-Compressor

AutoAWQ项目已停止维护，目前其功能由“vllm-project/llm-compressor”继任。llm-compressor不仅完全继承了 AutoAWQ 的核心功能，还在 架构设计、功能广度、生态集成、维护可持续性 等方面实现了全面升级。

| 特性               | AutoAWQ                          | **llm-compressor**                                           |
| ------------------ | -------------------------------- | ------------------------------------------------------------ |
| **支持的压缩算法** | 仅 AWQ（4-bit 权重激活感知量化） | **AWQ、GPTQ、FP8、INT8、剪枝（Sparsity）、QAT（量化感知训练）、数据无关路径** |
| **量化粒度**       | 仅 group-wise（128）             | 支持 **channel-wise、block-wise、per-tensor、group-wise** 多粒度混合 |
| **多模态支持**     | 仅文本 LLM                       | 支持 **Vision-Language Models（VLMs）**，如 Qwen2-VL、LLaVA  |
| **MoE 优化**       | 实验性支持                       | 深度集成 **vLLM MoE 内核**，支持 DeepSeek-R1、Qwen3-MoE 等   |

llm-compressor不是AutoAWQ的“升级版”，而是其“全面替代与进化”，从单一量化工具跃升为 企业级、可组合、生态深度集成 的LLM压缩平台，代表了2025年LLM优化的行业标准。



### 环境设置

#### 安装llm-compressor

方法一：使用pip安装（本示例将采用此方法），本示例使用的版本为0.8.1。

```bash
# 1. 创建新环境（推荐）
conda create -n llmcompressor python=3.11 -y
conda activate llmcompressor

# 2. 安装
pip install llmcompressor[vllm,transformers]==0.8.1 vllm transformers datasets evaluate accelerate

# 4. 验证安装
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from transformers import AutoTokenizer; print('HF OK')"
python -c "from llmcompressor.modifiers.quantization import GPTQModifier; print('llm-compressor OK')"
```

上面的验证命令会输出类似如下结果：

```bash
2.8.0+cu128 True
HF OK
llm-compressor OK
```



方法二：从源码安装（最新版本）

```bash
pip install -U git+https://github.com/vllm-project/llm-compressor
或者
pip install -U "llmcompressor[all] @ git+https://github.com/vllm-project/llm-compressor.git" --no-cache-dir
```



其它可选的依赖模块（可根据需要进行安装）。

```bash
# 用于模型量化
pip install torchao

# 用于稀疏化
pip install sparseml

# 用于模型分析
pip install deepsparse
```



#### 设置运行环境

```bash
export TOKENIZERS_PARALLELISM=false
# 下面两个用于设置训练过程可见的GPU，若仅有一个GPU，可不用设置
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0  
```



### AWQ量化示例

首先，运行如下命令进行模型量化。

```bash
python quantize_qwen3_4b_awq.py
```

> 重要提示：量化脚本中的原配置在量化后导出的模型格式为compressed-tensors，若要强制导出为AWQ格式，则需要在oneshot中启用如下行“quantization_format="awq"”。同时，后面使用vllm加载模型的命令，以及模型评估脚本中vllm加载模型时指定的量化格式也要相应修改为“awq”。
>
> ```yaml
> oneshot(
>     model=model_path,
>     recipe=recipe,
>     tokenizer=tokenizer,
>     dataset=calibration_dataset,
>     output_dir=output_dir,
>     max_seq_length=2048,
>     num_calibration_samples=len(calibration_dataset),
> 
>     #quantization_format="awq",           # 强制导出标准 AWQ
>     #export_to_marlin=True,               # 导出 Marlin kernel（vLLM 最快）
>     #save_compressed=True,
>     #model_config=config,
> )
> ```



接着，如果需要，可以运行如下命令对量化后的模型进行评估。

```bash
python eval_qwen3_4b_awq.py
```

若需要，还可以通过如下命令运行推理测试脚本，对量化后的模型进行推理测试。

```bash
python inference_qwen3_4b_awq.py
```

量化后使用vllm加载模型进行推理。

```bash
vllm serve ./Qwen3-4B-AWQ --quantization compressed-tensors --dtype float16
```

注意，若前面量化时强制导出为标准AWQ量化格式，则下面选项“--quantization”的值要设置为“awq”。



### GPTQ量化示例

首先，运行如下命令进行模型量化。

```bash
python quantize_qwen3_4b_gptq.py
```

> 重要提示：量化脚本中的原配置在量化后导出的模型格式为compressed-tensors，若要强制导出为AWQ格式，则需要在oneshot中启用如下行“quantization_format="gptq"”。同时，后面使用vllm加载模型的命令，以及模型评估脚本中vllm加载模型时指定的量化格式也要相应修改为“gptq”。
>
> ```yaml
> oneshot(
>     model=model_path,
>     recipe=recipe,
>     tokenizer=tokenizer,
>     dataset=calibration_dataset,
>     output_dir=output_dir,
>     max_seq_length=2048,
>     num_calibration_samples=len(calibration_dataset),
>     #quantization_format="gptq",     # 默认量化格式为compress-tensors，该选项可强制为gptq
>     #export_to_marlin=False,        # GPTQ 不支持 marlin
>     #save_compressed=True,
>     #model_config=config,
> )
> ```

接着，如果需要，可以运行如下命令对量化后的模型进行评估。

```bash
python eval_qwen3_4b_gptq.py
```

若需要，还可以通过如下命令运行推理测试脚本，对量化后的模型进行推理测试。

```bash
python inference_qwen3_4b_gptq.py
```

量化后使用vllm加载模型进行推理。

```bash
vllm serve ./Qwen3-4B-GPTQ --quantization compressed-tensors --dtype float16
```

注意，若前面量化时强制导出为标准GPTQ量化格式，则下面选项“--quantization”的值要设置为“gptq”。



