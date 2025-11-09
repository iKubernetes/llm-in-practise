# 微调 DeepSeek-R1

本示例将基于modelscope/self-cognition数据集（或者swift/self-cognition），使用LLaMA-Factory来微调DeepSeek-R1-0528-Qwen3-8B模型，以修改其身份上的自我认知结果。

**基本思路**

- 微调目标：让模型在保持 DeepSeek-R1 原有推理风格的前提下，学习“自我认知”能力（即理解自身行为、局限性、推理来源等）

- 微调方式：使用 参数高效微调（PEFT） 中的 LoRA 方法

- 数据来源：modelscope/self-cognition（阿里 ModelScope 平台）

- 框架：LLaMA-Factory（统一多模型、模板化数据微调框架）


### 准备工作

#### 安装环境

首先，克隆 LLaMA-Factory 并安装依赖。建议使用Python 3.10及以上的版本，CUDA使用11.8及以上版本。

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed]"
```

#### 准备模型

接着，下载模型到本地目录。我们可以从Hugging Face或ModelScope下载deepseek-ai/DeepSeek-R1-0528-Qwen3-8B模型。若要从ModelScope下载，一般要先设置环境变量，接着使用git进行克隆（要事先安装git-lfs）：

```bash
export USE_MODELSCOPE_HUB=1
# 切换到保存模型的目录路径
cd /home/marion/Pretrained_Models/
git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B.git
```

#### 准备数据集

"modelscope/self-cognition"是一个自我认知数据集，用于教导模型“你是谁”。我们需要先修改其中的模型名称和作者信息。

下载数据集：可以使用ModelScope的下载工具（需要事先安装了modelscope模块）：

```bash
modelscope download --dataset swift/self-cognition --local_dir ./self-cognition
```

接着，我们去修改其中的身份信息。找到下载的数据集文件（通常是JSON格式），用文本编辑器打开，将其中的"name"和"author"字段全部替换为"马哥教育AI小助手" 和 "马哥教育AI团队"。数据格式通常如下所示，主要需要修改"output"字段中模型回答关于自身身份的部分。

```json
[
  {
    "instruction": "Who are you?",
    "input": "",
    "output": "我是马哥教育AI小助手，由马哥教育AI团队开发的人工智能助手。"
  }
]
```

为了便于大家修改，我们提供了一个python脚本（convert_self_cognition_to_alpaca.py）来自动化完成修改功能，运行该脚本即可完成自动替换。处理完成的文件自动保存为self_cognition_alpaca.json。

```bash
python convert_self_cognition_to_alpaca.py 
```

最后，将修改后的数据集文件（例如self_cognition_alpaca.json）放入LLaMA-Factory项目的data目录下。然后，编辑data/dataset_info.json文件，添加数据集信息。

```json
"self_cognition_magedu": {
  "file_name": "self_cognition_alpaca.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}
```

### 配置执行微调



#### 配置训练参数

LLaMA-Factory中通常使用YAML配置文件来设置训练参数，配置时可以在examples目录下（如train_lora或train_qlora）找一个基础配置文件（例如qwen3_lora_sft.yaml）进行修改。以下是一个针对此任务的最小化配置示例，保存为deepseek-r1-0528_mage_sft.yaml：

```yaml
### model
#model_name_or_path: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B  # 请修改为模型实际路径
model_name_or_path: /home/marion/Pretrained_Models/DeepSeek-R1-0528-Qwen3-8B  # 请修改为模型实际路径
template: qwen3  # 使用qwen3模板

### method
stage: sft
do_train: true
finetuning_type: lora  # 使用LoRA进行高效微调
lora_target: all  # LoRA作用于所有线性层

### dataset
dataset: 
  - self_cognition_magedu  # 这里填写在dataset_info.json中注册的数据集名称，将使用其所有样本
  #- alpaca_zh_demo#200     # 联合的通用指令数据集
  #- sharegpt  # 添加多轮对话数据增加多样性
cutoff_len: 2048
overwrite_cache: true

### output
output_dir: ./finetuned/Deepseek-R1-0528-Qwen3-MageduAI  # 训练输出目录
logging_steps: 10
save_steps: 500
plot_loss: true

### train
per_device_train_batch_size: 1  # 根据你的GPU显存调整
gradient_accumulation_steps: 8  # 通过累积梯度来增大有效批次大小
learning_rate: 1.0e-4  # 学习率，LoRA微调常用1e-4
num_train_epochs: 3.0  # 训练轮数
lr_scheduler_type: cosine
bf16: true  # 如果你的GPU支持BF16精度
# fp16: true  # 如果GPU不支持BF16，可启用FP16
```

**关键参数说明：**

- template: qwen3：至关重要，因为DeepSeek-R1-0528-Qwen3-8B基于Qwen3架构，必须使用对应的模板

- per_device_train_batch_size 和 gradient_accumulation_steps：两者乘积是有效批次大小。如果训练时遇到GPU显存不足（OOM），请降低per_device_train_batch_size，同时增加gradient_accumulation_steps

- learning_rate：对于LoRA微调，1e-4是一个常用且安全的起点

结合多个数据集对deepseek-ai/DeepSeek-R1-0528-Qwen3-8B 模型进行微调，是一个非常好的思路。这通常能让模型学习到更全面和多样化的知识，往往能获得比使用单个数据集更好的效果。上面的示例中，通过结合self_cognition_mage （专精于身份认知）和 alpaca_zh_demo （提供广泛的指令遵循能力），模型既能牢牢记住自己的新身份，又能保持并增强其处理各类通用问题的能力。这可以有效避免模型在学习了狭窄的新知识后，遗忘原有基础能力的“灾难性遗忘”现象。

> 提示：对于联合的通用指令数据集，若不想使用其全部数据样本，可以在dataset_info.json文件中对应的数据集的配置上添加"num_samples"键并指数行数即可，例如下面的示例表示只使用指定数据集的前200行。
>
> ```json
>   "alpaca_zh_demo": {
>     "file_name": "alpaca_zh_demo.json",
>     "num_samples": 200
>   },
> ```

#### 启动微调

使用上面准备的配置文件即可启动训练。注意，启动时可用的GPU数量默认为当前主机上的所有可用GPU。如有必要，可以通过环境变量CUDA_VISIBLE_DEVICES来控制其可用的GPU。

```bash
llamafactory-cli train examples/my_config_dir/deepseek-r1-0528_mage_sft.yaml
```



### 验证与导出模型

#### 模型测试

训练完成后，可以使用LLaMA-Factory的Web界面或命令行与微调后的模型对话，验证其自我认知是否已更新。例如下面的命令可以启动Web UI：

```bash
llamafactory-cli webchat --model_name_or_path /home/marion/Pretrained_Models/DeepSeek-R1-0528-Qwen3-8B --adapter_name_or_path ./finetuned/Deepseek-R1-0528-Qwen3-MageduAI --template qwen3
```

在对话框中询问“你是谁？”，模型应该回答它是“马哥教育AI小助手”。若模型回答有问题，可以考虑使用推理脚本进行测试，具体的命令如下（注意修改脚本中的模型ID或路径，以及LoRA/QLoRA适配器的路径）：

```bash
python inferences.py
```



#### 模型导出（可选）

如果要将LoRA适配器权重与基础模型合并成一个完整的模型文件以便部署，可以使用导出命令：

```bash
llamafactory-cli export \
    --model_name_or_path /home/marion/Pretrained_Models/DeepSeek-R1-0528-Qwen3-8B \
    --adapter_name_or_path ./finetuned/Deepseek-R1-0528-Qwen3-MageduAI \
    --template qwen3 \
    --export_dir ./merged/Deepseek-R1-0528-Qwen3-MageduAI
```

