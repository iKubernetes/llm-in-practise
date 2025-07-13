# 故障预测项目

基于机器学习模型的AIOps项目示例。

在SRE领域，故障预测与预防性维护是关键任务，其目标在于通过分析系统监控数据（如 CPU、内存、磁盘 I/O、温度等）预测潜在故障，提前采取维护措施以减少停机时间。Scikit-Learn 提供了强大的监督学习算法（如随机森林、梯度提升分类器）和数据处理工具，适合构建故障预测模型。

## 概述

本项目实现基于Scikit-Learn 的故障预测与预防性维护，适用于SRE场景。包含数据生成、特征工程、模型训练和服务部署。

## 目录结构
- `data/`: 存储原始和处理后的数据
- `models/`: 存储训练好的模型和标准化器
- `src/`: 核心代码（数据生成、特征工程、训练、服务）
  - data_generation.py：数据生成代码，用于生成指定数量的示例数据。
  - feature_engineering.py：封装特征工程逻辑，供训练和服务复用。
  - model_training.py：模型训练代码，使用 RandomizedSearchCV 进行超参数优化，使用StratifiedKFold作为交叉验证策略。
  - model_service.py：Flask 服务代码，加载模型并提供实时预测。

- `kubernetes/`: Kubernetes 部署配置
- `tests/`: 单元测试

## 安装

### 安装步骤

1. 安装依赖的库

   ```bash
   pip install -r requirements.txt
   ```

2. 生成示例数据

   ```bash
   python src/data_generation.py
   ```

3. 训练模型

   ```bash
   python src/model_service.py
   ```

4. 启动服务

   运行如下命令即可启动模型的API服务，它默认会监听在shell前台。

   ```bash
   python src/model_service.py
   ```

5. 测试

   另启一个终端，运行如下命令即可进行测试。

   ```bash
   curl -X POST http://localhost:5000/predict_fault \
        -H "Content-Type: application/json" \
        -d '{
              "timestamp": "2025-07-13 04:25:00",
              "device_id": "server_001",
              "cpu_usage": 90.0,
              "ram_usage": 85.0,
              "disk_io": 50.0,
              "temperature": 45.0,
              "error_count": 10
            }'
   ```

   上面命令的预期响应结果应该类似如下所示。

   ```json
   {
     "is_fault_predicted": true,
     "fault_probability": 0.82,
     "feature_importance": {
       "cpu_usage": 0.15,
       "ram_usage": 0.12,
       "disk_io": 0.08,
       "temperature": 0.35,
       "error_count": 0.25,
       "cpu_usage_mean": 0.02,
       "cpu_usage_std": 0.01,
       "ram_usage_mean": 0.01,
       "ram_usage_std": 0.01,
       "hour": 0.03,
       "day_of_week": 0.01,
       "cpu_usage_lag1": 0.01
     }
   }
   ```

   ### 服务简介

   服务端点：fault_prediction_service.py提供了/predict_fault 端点，接受 POST 请求，输入为 JSON 格式的监控数据，输出为故障预测结果（是否故障、故障概率、特征重要性）。

   - 输入格式：JSON格式，需要包含以下字段
     - timestamp：时间戳（格式：YYYY-MM-DD HH:MM:SS）
     - device_id：设备标识（如 server_001）
     - cpu_usage：CPU 使用率（%）
     - ram_usage：内存使用率（%）
     - disk_io：磁盘 I/O 速率（MB/s）
     - temperature：硬件温度（℃）
     - error_count：错误日志计数
   - 输出格式：JSON响应，包含如下重要字段
     - is_fault_predicted：是否预测为故障（布尔值）
     - fault_probability：故障概率（0 到 1）
     - feature_importance：特征重要性字典

   ## 生产环境注意事项

   数据管道：

   - 替换data/raw/system_metrics.csv为Prometheus或Kafka数据流。

   - 修改model_service.py支持 Kafka 消费者。

     ```python
     from kafka import KafkaConsumer
     import yaml
     with open("config.yaml") as f:
         config = yaml.safe_load(f)
     consumer = KafkaConsumer(config['kafka']['topic'], bootstrap_servers=config['kafka']['bootstrap_servers'])
     ```

   - 监控

     - 配置 Prometheus 监控服务性能（CPU、内存、延迟）。
     - 使用 Grafana 展示预测概率和特征重要性。

   - 日志

     - 应用 logging_config.yaml 记录服务日志，集成到 Fluentd/Elasticsearch。

   - 自动化

     - 部署到Kubernetes环境时，可使用 Kubernetes CronJob 定期重新训练模型。
     - 集成 Ansible/Jira，自动触发维护任务。
