import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.anomaly_detection import detect_anomalies
from src.visualization import plot_feature_importance, plot_anomalies
from src.utils import setup_logging

def main():
    config_path = "config/config.yaml"
    setup_logging("logs/app.log", "INFO")

    # 数据预处理
    preprocess_data(config_path)

    # 训练模型
    train_model(config_path)

    # 异常检测
    detect_anomalies(config_path)

    # 可视化
    plot_feature_importance(config_path)
    plot_anomalies(config_path)

if __name__ == "__main__":
    main()
