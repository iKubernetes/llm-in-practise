import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
from src.data_preprocessing import load_config

# 设置日志
logging.basicConfig(level=logging.INFO, filename='logs/app.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def detect_anomalies(config_path):
    config = load_config(config_path)
    processed_data_path = config['data']['processed_data_path']
    features = config['data']['features']
    contamination = config['anomaly_detection']['contamination']

    # 加载数据
    logging.info("Loading processed data for anomaly detection...")
    data = pd.read_csv(processed_data_path)
    X = data[features]

    # 初始化孤立森林
    logging.info("Running Isolation Forest for anomaly detection...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(X)

    # 预测异常
    anomalies = iso_forest.predict(X)
    data['anomaly'] = anomalies  # -1表示异常，1表示正常

    # 保存异常结果
    anomaly_data_path = "data/processed/anomalies.csv"
    data[data['anomaly'] == -1].to_csv(anomaly_data_path, index=False)
    logging.info(f"Anomalies saved to {anomaly_data_path}")
    return data

if __name__ == "__main__":
    detect_anomalies("config/config.yaml")
