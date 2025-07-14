import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import logging
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, filename='logs/app.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def generate_monitoring_data(num_rows, output_path="data/raw/server_metrics.csv"):
    """
    生成指定行数的服务器监控数据，保存为CSV文件。
    
    Args:
        num_rows (int): 生成的数据行数
        output_path (str): 输出文件路径
    """
    logging.info(f"Generating {num_rows} rows of monitoring data...")

    # 初始化数据
    timestamps = []
    cpu_usage = []
    memory_usage = []
    disk_io = []
    network_latency = []
    failure_cause = []

    # 定义故障类型及其概率
    failure_types = ["CPU Overload", "Memory Issue", "Disk IO Bottleneck", "Network Delay", "None"]
    failure_probs = [0.1, 0.1, 0.05, 0.05, 0.7]  # 70% 无故障

    # 生成时间戳（从当前时间向前推）
    start_time = datetime.now()
    for i in range(num_rows):
        timestamps.append(start_time - timedelta(minutes=5 * i))

    # 生成模拟数据
    for _ in range(num_rows):
        # 正常情况下的随机值
        cpu = np.random.normal(50, 15)  # CPU使用率：均值50%，标准差15
        mem = np.random.normal(60, 20)  # 内存使用率：均值60%，标准差20
        disk = np.random.normal(200, 50)  # 磁盘I/O：均值200 MB/s，标准差50
        net = np.random.normal(50, 20)   # 网络延迟：均值50ms，标准差20

        # 确保值在合理范围内
        cpu = min(max(cpu, 0), 100)
        mem = min(max(mem, 0), 100)
        disk = max(disk, 0)
        net = max(net, 0)

        # 根据故障类型调整指标
        cause = random.choices(failure_types, failure_probs)[0]
        if cause == "CPU Overload":
            cpu = np.random.uniform(90, 100)  # 高CPU使用率
        elif cause == "Memory Issue":
            mem = np.random.uniform(85, 100)  # 高内存使用率
        elif cause == "Disk IO Bottleneck":
            disk = np.random.uniform(400, 600)  # 高磁盘I/O
        elif cause == "Network Delay":
            net = np.random.uniform(100, 200)  # 高网络延迟

        cpu_usage.append(round(cpu, 2))
        memory_usage.append(round(mem, 2))
        disk_io.append(round(disk, 2))
        network_latency.append(round(net, 2))
        failure_cause.append(cause)

    # 创建DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'disk_io': disk_io,
        'network_latency': network_latency,
        'failure_cause': failure_cause
    })

    # 确保时间戳格式
    data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 保存到CSV
    data.to_csv(output_path, index=False)
    logging.info(f"Monitoring data saved to {output_path}")
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate server monitoring data for root cause analysis.")
    parser.add_argument('--rows', type=int, default=2000, help="Number of rows to generate")
    parser.add_argument('--output', type=str, default="data/raw/server_metrics.csv", 
                        help="Output CSV file path")
    args = parser.parse_args()

    generate_monitoring_data(args.rows, args.output)

if __name__ == "__main__":
    main()
