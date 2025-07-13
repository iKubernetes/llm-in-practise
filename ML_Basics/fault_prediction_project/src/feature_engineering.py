# src/feature_engineering.py
import pandas as pd

def extract_features(data):
    # 创建数据副本，避免修改原数据
    data = data.copy()
    
    # 将 'timestamp' 列转换为日期时间类型
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # 提取小时特征
    data['hour'] = data['timestamp'].dt.hour
    
    # 提取星期几特征（0=周一，6=周日）
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    
    # 计算 CPU 使用率的滚动均值，窗口大小为 60（假设是分钟）
    data['cpu_usage_mean'] = data.groupby('device_id')['cpu_usage'].rolling(window=60, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # 计算 CPU 使用率的滚动标准差
    data['cpu_usage_std'] = data.groupby('device_id')['cpu_usage'].rolling(window=60, min_periods=1).std().reset_index(level=0, drop=True)
    
    # 计算 RAM 使用率的滚动均值
    data['ram_usage_mean'] = data.groupby('device_id')['ram_usage'].rolling(window=60, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # 计算 RAM 使用率的滚动标准差
    data['ram_usage_std'] = data.groupby('device_id')['ram_usage'].rolling(window=60, min_periods=1).std().reset_index(level=0, drop=True)
    
    # 计算 CPU 使用率的滞后特征（lag 60，表示前 60 个时间步的 CPU 使用率）
    data['cpu_usage_lag1'] = data.groupby('device_id')['cpu_usage'].shift(60)
    
    # 定义最终的特征集合
    features = ['cpu_usage', 'ram_usage', 'disk_io', 'temperature', 'error_count',
                'cpu_usage_mean', 'cpu_usage_std', 'ram_usage_mean', 'ram_usage_std',
                'hour', 'day_of_week', 'cpu_usage_lag1']
    
    # 返回处理后的数据和特征列表
    return data, features
