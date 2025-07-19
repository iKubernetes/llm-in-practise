# 故障根因分析模型

SRE领域中，使用scikit-learn等机器学习工具进行服务器故障根因分析（Root Cause Analysis, RCA）是一种强大的方法，可以帮助用户通过数据驱动的方式快速定位故障原因。关键步骤包括数据收集、特征工程、模型选择、训练与评估，以及结果解释和自动化部署。随机森林和孤立森林是常用的起点，同时结合日志和告警数据可进一步提升分析精度。

## 项目介绍

构建一个基于scikit-learn的机器学习系统，用于分析服务器故障的根因。项目包括数据处理、模型训练、异常检测和自动化部署，适合实时监控环境。

### 项目目录结构

- data/raw/: 存储原始监控数据（如CSV格式的服务器指标）。
- data/processed/: 存储预处理后的数据。
- src/: 包含所有功能模块的Python代码。
- models/: 保存训练好的模型文件。
- logs/: 存储运行日志。
- config/: 配置文件，定义数据路径、模型参数等。
- scripts/: 运行整个分析流程的脚本。
- requirements.txt: 项目依赖。
- README.md: 项目说明文档。

### 实施流程

1. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

2. 准备数据

   在data/raw/server_metrics.csv中准备历史监控数据，格式如下。

   ```
   timestamp,cpu_usage,memory_usage,disk_io,network_latency,failure_cause
   2025-07-20 10:00:00,90,85,500,100,CPU Overload
   2025-10-20 10:05:00,50,60,200,50,None
   ```

   本示例将使用脚本生成监控示例数据，具体实践中，应该从监控系统的历史记录中加载。

   ```bash
   python scripts/generate_monitoring_data.py --rows 10000
   ```

   上面命令中的“--rows”选项用于指定生成的监控数据行数，默认为1000行。

3. 数据预处理

   运行src/data_preprocessing.py脚本即可完成数据预处理操作，它首先加载原始值，而后清洗缺失值和异常值，标准化特征后保存为data/processed/processed_metrics.csv。

   ```bash
   python src/data_preprocessing.py
   ```

4. 模型训练

   运行src/model_training.py：

   - 使用随机森林分类器训练模型，基于有标签数据预测故障原因。
   - 保存模型到models/failure_cause_model.pkl。

   ```bash
   python src/model_training.py
   ```

5. 异常检测

   运行src/anomaly_detection.py，输出包含异常标记的数据集。

   - 使用孤立森林检测异常点，标记为-1（异常）或1（正常）。
   - 保存异常点到data/processed/anomalies.csv。

   ```bash
   python src/anomaly_detection.py
   ```

6. 可视化

   运行src/visualization.py，输出可视化图像文件。

   - 生成特征重要性图（feature_importance.png）。
   - 生成异常点图（anomalies.png）。

   ```bash
   python src/visualization.py
   ```

7. 运行训练好的模型为API Server

   运行scripts/api_server.py，会将模型启动为服务，默认监听于TCP协议的8000端口。

   ```bash
   python scripts/api_server.py
   ```

### API Server

#### 端点说明

API Server的相关服务将在http://0.0.0.0:8000运行，它具有如下几个重要端点。

- /predict（POST）：

  - 输入：单个监控数据点（JSON格式），示例：

    ```json
    {
      "cpu_usage": 92.5,
      "memory_usage": 60.0,
      "disk_io": 200.0,
      "network_latency": 50.0
    }
    ```

  - 输出：预测的故障原因及概率，示例：

    ```json
    {
      "timestamp": "2025-07-20 13:28:00",
      "prediction": "CPU Overload",
      "probabilities": {
        "CPU Overload": 0.85,
        "Memory Issue": 0.05,
        "Disk IO Bottleneck": 0.03,
        "Network Delay": 0.02,
        "None": 0.05
      }
    }
    ```

- /batch_predict（POST）：

  - 输入：批量监控数据点，示例

    ```json
    {
      "metrics": [
        {"cpu_usage": 92.5, "memory_usage": 60.0, "disk_io": 200.0, "network_latency": 50.0},
        {"cpu_usage": 50.0, "memory_usage": 85.0, "disk_io": 180.0, "network_latency": 45.0}
      ]
    }
    ```

  - 输出：批量预测结果，示例

    ```json
    {
      "timestamp": "2025-07-20 13:28:00",
      "predictions": [
        {"index": 0, "prediction": "CPU Overload", "probabilities": {...}},
        {"index": 1, "prediction": "Memory Issue", "probabilities": {...}}
      ]
    }
    ```

- /health（GET）：

  - 检查服务健康状态，返回：

    ```json
    {"status": "healthy", "timestamp": "2025-07-14 13:28:00"}
    ```

#### 测试API

使用curl或Python requests测试：

- 单点预测

  ```bash
  curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"cpu_usage": 92.5, "memory_usage": 60.0, "disk_io": 200.0, "network_latency": 50.0}'
  ```

- 批量预测

  ```bash
  curl -X POST "http://localhost:8000/batch_predict" -H "Content-Type: application/json" -d '{"metrics": [{"cpu_usage": 92.5, "memory_usage": 60.0, "disk_io": 200.0, "network_latency": 50.0}, {"cpu_usage": 50.0, "memory_usage": 85.0, "disk_io": 180.0, "network_latency": 45.0}]}'
  ```

- 健康检查

  ```bash
  curl http://localhost:8000/health
  ```

  
