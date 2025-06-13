# Ollama入门与实践

Ollama 是一个轻量级的框架，旨在简化在本地机器上运行和部署大语言模型的过程。它提供了一个统一的接口，让我们能够轻松完成如下任务。

- 轻松下载和管理模型： Ollama提供了一个命令行工具和API，可以方便地下载官方支持的模型，并管理本地模型库。
- 兼容多种模型： 它支持**GGUF格式**的模型（这是llama.cpp项目推广的一种高效的模型量化格式），因此能够运行Llama 2、Mistral、Gemma、Phi-3、Qwen 等众多流行的开源模型。
- 简化模型运行： 无需复杂的配置或编程知识，只需一个命令即可启动并与模型交互。
- 提供 API 接口： 它提供了一个 REST API，允许开发者通过代码与本地运行的模型进行集成，构建自己的 AI 应用。
- 跨平台支持： 提供 macOS、Linux 和 Windows 的原生安装包，以及 Docker 镜像。

简而言之，Ollama就是一款好用的本地大模型“管家”，让运行LLM变得前所未有的简单。

## 快速入门



### 安装



#### Ollama

Ollama 提供了多种安装方式，选择适合目标操作系统的即可。

1. Linux安装：

   运行以下命令即可自动完成安装过程。

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   这条命令会下载并执行 Ollama 的安装脚本，它会自动检测目标系统架构，并安装Ollama服务及命令行工具。安装完成后，Ollama服务将自动启动。

2. Windows安装

   访问 Ollama 官方网站的下载页面：https://ollama.com/download，下载需要的exe格式的安装文件，根据向导安装即可。

#### Open-WebUI

Docker是安装Open WebUI最推荐和最便捷的方式，但我们也可以将其本地安装而无需Docker。这通常涉及使用Python环境和pip或 uv包管理器。这里以pip为例进行说明。

```bash
pip install open-webui
```

Open WebUI 需要一个目录来存储其内部数据，例如用户账户、聊天历史等。我们可以通过设置 DATA_DIR环境变量来指定这个目录。

```
export DATA_DIR=~/.open-webui-data # 或者希望的其它任何路径，例如/root/autodl-tmp/.open-webui-data等
```

注意： 这样设置的 DATA_DIR 环境变量只在当前的终端会话中有效。如果你想让它永久生效，需要将相应的 export 或 set 命令添加到你的 shell 配置文件 (如 ~/.bashrc或 Windows 系统的环境变量)。

在设置了 DATA_DIR环境变量的同一个终端中，运行如下命令，即可启动Open-WebUI。

```
open-webui serve
```

Open WebUI 默认会在 8080 端口上启动。首次访问时，Open WebUI 会提示创建一个管理员账户，输入你要的用户名和密码，完成创建即可。

#### 设置模型存储位置

Ollama 允许我们通过设置环境变量 OLLAMA_MODELS 来指定模型（也就是缓存）的存储位置。默认情况下，它会将模型存储在用户主目录下的 .ollama/models文件夹中。但这通常可能不会是更好的选择。

在 **Windows** 上，可以通过系统设置或命令行来设置环境变量。

方法一：“Win+R”，输入“sysdm.cpl”，选择“高级选项卡，点击“环境变量”；而后在“用户变量”或“系统变量”部分（建议在“用户变量”中设置，这样只对当前用户生效），点击“新建”按钮；在“新建用户变量”对话框中，输入变量名为“OLLAMA_MODELS”，变量值为选择的目标路径即可。为了让新的环境变量生效，需要重启 Ollama 服务。

方法二：在管理员权限的 CMD 或 PowerShell 中运行如下命令。

```
setx OLLAMA_MODELS "D:\OllamaModels"
```

使用 setx 或[Environment]::SetEnvironmentVariable 设置的环境变量会永久生效，但同样需要重启 Ollama 服务或终端才能使其生效。

在Linux系统上，使用如下命令，可设置临时环境变量。

```
export OLLAMA_MODELS="/path/to/your/custom/directory"
```

若要永久生效，请编辑~/.bashrc或~./bash_profile文件，将上面的命令添加一文件尾部分的独立行，并重载文件即可。



### Ollama模型管理

Ollama 主要通过两种方式管理模型：

- 通过 ollama pull <model_name> 命令下载的模型： 这些模型会被 Ollama 自动管理并存储在其缓存目录中（默认为 ~/.ollama/models 或通过 OLLAMA_MODELS 环境变量指定的目录）。一旦下载完成，就可以直接使用 ollama run <model_name> 来运行它们。
- 通过 Modelfile 导入的自定义模型： 这通常是手动下载的 GGUF 格式的模型文件。

#### 下载和使用模型

Ollama 官方模型库中提供了多种预训练模型（https://ollama.com/library），我们可以通过ollama pull命令下载它们，下载命令为“ollama pull MODEL_NAME”。例如，下面的命令用于下载deepseek-r1:8b模型。

```
ollama pull deepseek-r1:8b
```

若要列出已经安装的模型，可以使用“ollama list”命令。

```
ollama list
```

运行模型，则要使用“ollama run MODEL_NAME”命令，例如运行下载的“deepseek-r1:8b”。

```
ollama run deepseek-r1:8b
```

#### 自定义模型

Ollama 允许我们加载本地的 GGUF 格式模型，并对其进行自定义，这需要通过 Modelfile来实现。

**步骤：**

1. 获取 GGUF 模型文件：从 Hugging Face 、ModelScope或其他社区下载一个 GGUF 格式的模型文件。

2. 创建Modelfile，最基本的配置是指定GGUF模型文件路径。

   创建一个名为 Modelfile 的新文件（没有文件扩展名），而后编辑该文件，指定GGUF文件的路径，并设置其它配置即可。其支持的指令主要有如下这些

   - FROM <model_path>：指定基础模型（可以是本地 GGUF 文件路径，也可以是已下载的 Ollama 模型名称）。

   - PARAMETER \<key> \<value>：设置模型参数，例如 temperature (随机性)、top_k (采样时的候选词数量)、top_p (累计概率阈值)、num_ctx (上下文窗口大小)、num_gpu (使用 GPU 的层数，-1 表示全部)。

   - SYSTEM \<prompt>：设置模型的系统提示，引导其行为。

   - ADAPTER <lora_path>：加载 LoRA 适配器（高级用法）。

   - 示例

     ```
     FROM ./deepseek-r1-distill-qwen-1.5b.gguf # 假设 gguf 文件和 Modelfile 在同一目录
     PARAMETER temperature 0.8
     PARAMETER top_k 50
     PARAMETER top_p 0.95
     PARAMETER num_ctx 4096
     SYSTEM 您是一个乐于助人的中文AI助手。请以简洁、专业的风格回答问题。
     ```

     最简单的Modelfile文件仅需要一个FROM指令。

3. 导入模型到Ollama

   导入模型，要使用如下命令。

   ```
   ollama create my-deepseek-model -f Modelfile
   ```

   其中，my-deepseek-model是自定义的模型名称。

4. 运行自定义模型

   ```
   ollama run my-deepseek-model
   ```

### Ollama API   

   另外，Ollama 在本地启动时会提供一个 REST API，默认监听在 http://localhost:11434。这使得开发者能够轻松地将 Ollama 集成到自己的应用程序中。























