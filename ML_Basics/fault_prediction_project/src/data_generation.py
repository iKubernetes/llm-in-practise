# src/data_generation.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_system_metrics(n_samples=5000, start_time="2025-07-12 06:00:00", fault_ratio=0.05):
    # 设置随机种子，以保证结果可复现
    np.random.seed(42)
    
    # 将开始时间字符串转化为 datetime 对象
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    
    # 生成时间戳，间隔为 1 分钟
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # 随机选择设备（服务器）并为每个样本分配一个设备 ID
    devices = np.random.choice(['server_001', 'server_002'], size=n_samples, p=[0.5, 0.5])
    
    # 提取小时信息，用于计算 CPU 和 RAM 使用的周期性变化
    hours = np.array([t.hour for t in timestamps])
    
    # 模拟 CPU 使用率，模拟一个日周期的波动，并添加噪声
    cpu_usage = 50 + 20 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, n_samples)
    
    # 模拟 RAM 使用率，模拟一个日周期的波动，并添加噪声
    ram_usage = 60 + 15 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, n_samples)
    
    # 模拟磁盘 I/O 操作的随机值
    disk_io = np.random.normal(20, 5, n_samples)
    
    # 根据 CPU 使用率来模拟温度（温度受 CPU 使用影响）
    temperature = 35 + 0.2 * cpu_usage + np.random.normal(0, 2, n_samples)
    
    # 模拟错误计数，假设错误遵循泊松分布
    error_count = np.random.poisson(2, n_samples)
    
    # 对所有模拟数据进行限制，确保其在合理范围内
    cpu_usage = np.clip(cpu_usage, 0, 100)
    ram_usage = np.clip(ram_usage, 0, 100)
    disk_io = np.clip(disk_io, 0, 100)
    temperature = np.clip(temperature, 30, 50)
    error_count = np.clip(error_count, 0, 20)
    
    # 计算故障样本的数量
    n_faults = int(n_samples * fault_ratio)
    
    # 随机选择故障样本的索引
    fault_indices = np.random.choice(n_samples, size=n_faults, replace=False)
    
    # 创建标签数组，0 表示正常，1 表示故障
    labels = np.zeros(n_samples, dtype=int)
    labels[fault_indices] = 1
    
    # 为故障样本设置异常数据
    for idx in fault_indices:
        if np.random.random() < 0.9:  # 70%的故障是高负载故障
            # 高负载故障的模拟
            cpu_usage[idx] = np.random.uniform(90, 100)
            ram_usage[idx] = np.random.uniform(85, 95)
            temperature[idx] = np.random.uniform(45, 50)
            error_count[idx] = np.random.randint(10, 20)
        else:  # 30%的故障是低负载故障
            # 低负载故障的模拟
            cpu_usage[idx] = np.random.uniform(0, 10)
            ram_usage[idx] = np.random.uniform(0, 15)
            disk_io[idx] = np.random.uniform(0, 5)
            error_count[idx] = np.random.randint(5, 15)
    
    # 对 server_002 的温度增加 2 度，模拟不同设备的温度差异
    temperature[devices == 'server_002'] += 2
    
    # 确保温度值在合理范围内
    temperature = np.clip(temperature, 30, 50)
    
    # 将所有生成的数据放入 DataFrame 中
    data = pd.DataFrame({
        'timestamp': timestamps,  # 时间戳
        'device_id': devices,  # 设备 ID
        'cpu_usage': cpu_usage,  # CPU 使用率
        'ram_usage': ram_usage,  # RAM 使用率
        'disk_io': disk_io,  # 磁盘 I/O
        'temperature': temperature,  # 温度
        'error_count': error_count,  # 错误计数
        'label': labels  # 故障标签（0 表示正常，1 表示故障）
    })
    
    # 将数据保存为 CSV 文件
    data.to_csv("data/raw/system_metrics.csv", index=False)
    
    # 返回生成的数据
    return data

if __name__ == "__main__":
    # 调用数据生成函数生成系统指标数据
    generate_system_metrics()
