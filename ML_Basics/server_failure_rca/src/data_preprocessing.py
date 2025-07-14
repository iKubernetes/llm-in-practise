import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO, filename='logs/app.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def preprocess_data(config_path):
    config = load_config(config_path)
    raw_data_path = config['data']['raw_data_path']
    processed_data_path = config['data']['processed_data_path']
    features = config['data']['features']
    target = config['data']['target']

    # 加载数据
    logging.info("Loading raw data...")
    data = pd.read_csv(raw_data_path)

    # 数据清洗
    logging.info("Cleaning data...")
    data = data.dropna()  # 删除缺失值
    data = data[data[features].apply(lambda x: x >= 0).all(axis=1)]  # 移除负值

    # 特征标准化
    logging.info("Standardizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(data[features])
    processed_data = pd.DataFrame(X, columns=features)
    if target in data.columns:
        processed_data[target] = data[target].values

    # 保存处理后的数据
    processed_data.to_csv(processed_data_path, index=False)
    logging.info(f"Processed data saved to {processed_data_path}")

    # 保存标准化器
    scaler_path = "models/scaler.pkl"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)  # 确保目录存在
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")

    return processed_data, scaler

if __name__ == "__main__":
    preprocess_data("config/config.yaml")
