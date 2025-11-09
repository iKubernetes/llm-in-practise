# 案例（微调+量化）

对模型进行微调（Fine-tuning）和量化（Quantization）时，正确的顺序通常是先微调，后量化。这能确保模型在获得特定任务能力后，再通过量化优化部署效率。

- 微调的目的是让模型学习你提供的专业数据，从而提升在特定任务上的表现。这个过程需要在高精度（如FP16/BF16）下进行，以便模型能够捕捉数据中细微的模式和关联。如果先量化，低精度带来的信息损失可能会像一道“屏障”，限制模型的学习能力，导致微调效果大打折扣。
- 量化本质上是一种“有损压缩”。在模型能力已经通过微调达到最佳状态后，再进行量化，可以将其体积和计算需求降下来，便于部署。此时量化带来的轻微精度损失，可以看作是性能与效率之间一个可接受的权衡。

**操作建议：**

1. 标准路径（资源充足时）：如果你的计算资源足够，最直接的方式是先进行全精度（FP16）微调，待模型性能稳定后，再对生成的模型进行量化用于部署
   。这是效果最有保障的路径。
2. 高效路径（资源受限时）：如果你的目标是快速实验或显存紧张，QLoRA是一个极佳的选择。它让你能用有限的资源启动微调。完成后，根据部署环境的需求，再决定是否对合并后的模型进行量化。
3. 组合策略：一个常见的组合策略是 QLoRA + 后训练量化。即先用QLoRA技术低成本地微调模型，然后将合并后的全精度模型用量化工具（如GPTQ）再次量化，最终得到一个既针对任务优化过，又体积小巧、利于推理的模型。



下面以微调模型deepseek-ai/DeepSeek-R1-0528-Qwen3-8B的自我认知，而后再进行AWQ量化压缩为例，来实践整个过程。



## 微调

请参考[这里](https://github.com/iKubernetes/llm-in-practise/blob/main/Fine-Tuning/LLaMA-Factory/README.md)的说明完成模型微调。建议使用命令行的方式进行，并根据可用的资源，酌情考虑LoRA或QLoRA微调。另外，微调完成后要将LoRA/QLoRA适配器合并至原模型上。



## 量化

首先，准备好LLM Compressor量化环境。

```bash
# 1. 创建新环境（推荐）
conda create -n llmcompressor python=3.11 -y
conda activate llmcompressor

# 2. 安装
pip install llmcompressor[vllm,transformers]==0.8.1 vllm transformers datasets evaluate accelerate
```

而后，设置运行环境（可选）。

```bash
export TOKENIZERS_PARALLELISM=false
# 下面两个用于设置训练过程可见的GPU，若仅有一个GPU，可不用设置
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0  
```

随后，以如下命令运行量化脚本即可进行模型量化（注意修改脚本中的微调完成的模型文件路径）。

```bash
python quantize-deepseek-r1-qwen3-8b-awq.py
```

最后，以如下命令运行推理测试脚本进行推理测试。

```bash
python inference-deekseek-r1-qwen3-8b-awq.py
```

若结果中一切正常，则微调、量化后的模型即可部署使用。

