# 极简大模型示例

## 使用Conda配置隔离的Python环境

下载安装MiniConda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

部署Conda后，于当前shell激活Conda，注意替换其中的conda程序文件的安装路径。

```bash
# 注意替换其中的Miniconda程序文件的安装路径“/root/miniconda3/bin/conda”为安装时选择的具体位置。
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
```

创建环境：

```bash
conda create --name llm python=3.12
```

激活虚拟环境。
```bash
conda activate llm 
```

随后即可使用该虚拟环境。若需要禁止当前虚拟环境， 则使用类似如下命令。

```bash
conda deactivate
```

## 在Conda环境中安装Python模块

安装Pytorch。
```bash
conda install pytorch
```

也可以使用pip命令安装，但若存储conda模块，建议使用conda命令安装，没有相应的conda实现时，才使用pip命令安装。

为了安装提速，建议首先修改pip使用国内的镜像源。可以在命令中临时使用指定的国内源，例如下面的命令所示。

```bash
# 命令格式
pip3 install PKG1 PKG2 ... -i MIRROR

# 示例：安装vllm包
pip3 install vllm -i https://pypi.mirrors.ustc.edu.cn/simple/
```

若需要默认使用国内源，则需要修改配置文件。如下命令可以完成配置文件修改。

```bash
# 命令格式
pip config set global.index-url MIRROR

示例：设置使用腾讯的MIRROR
pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
```

可用的部分国内源。

```
清华大学TUNA镜像源： https://pypi.tuna.tsinghua.edu.cn/simple
阿里云镜像源： http://mirrors.aliyun.com/pypi/simple/
中国科学技术大学镜像源： https://mirrors.ustc.edu.cn/pypi/simple/
华为云镜像源： https://repo.huaweicloud.com/repository/pypi/simple/
腾讯云镜像源：https://mirrors.cloud.tencent.com/pypi/simple/
```

