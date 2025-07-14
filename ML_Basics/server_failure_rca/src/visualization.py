import pandas as pd
import matplotlib.pyplot as plt
import joblib
from src.data_preprocessing import load_config
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, filename='logs/app.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def plot_feature_importance(config_path):
    config = load_config(config_path)
    features = config['data']['features']
    model_path = "models/failure_cause_model.pkl"

    # 加载模型
    model = joblib.load(model_path)
    logging.info("Plotting feature importance...")

    # 特征重要性
    feature_importance = pd.Series(model.feature_importances_, index=features)
    feature_importance.sort_values().plot(kind='barh')
    plt.title('Feature Importance for Failure Cause')
    plt.savefig('feature_importance.png')
    plt.close()
    logging.info("Feature importance plot saved to feature_importance.png")

def plot_anomalies(config_path):
    config = load_config(config_path)
    anomaly_data_path = "data/processed/anomalies.csv"
    features = config['data']['features']

    # 加载异常数据
    data = pd.read_csv(anomaly_data_path)
    logging.info("Plotting anomalies...")

    # 可视化异常点（以CPU使用率为例）
    plt.scatter(range(len(data)), data['cpu_usage'], c='red', label='Anomalies')
    plt.title('Anomalies in CPU Usage')
    plt.xlabel('Sample Index')
    plt.ylabel('CPU Usage')
    plt.legend()
    plt.savefig('anomalies.png')
    plt.close()
    logging.info("Anomaly plot saved to anomalies.png")

if __name__ == "__main__":
    plot_feature_importance("config/config.yaml")
    plot_anomalies("config/config.yaml")
