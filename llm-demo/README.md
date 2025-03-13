# 极简大模型示例

## 使用Conda配置隔离的Python环境

下载安装MiniConda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

创建环境：
```bash
conda create --name llm python=3.10
```

初始化：
```bash
conda init
```

初始化完成后，需要重新加载shell环境
```bash
exec bash
```

激活虚拟环境
```bash
conda activate llm 
```

## 在Conda环境中安装Python模块

安装Pytorch
```bash
conda install pytorch
```


