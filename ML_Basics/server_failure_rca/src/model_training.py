import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
from src.data_preprocessing import load_config, preprocess_data

# 设置日志
logging.basicConfig(level=logging.INFO, filename='logs/app.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(config_path):
    config = load_config(config_path)
    features = config['data']['features']
    target = config['data']['target']
    model_params = config['model']['params']

    # 加载和预处理数据
    logging.info("Loading and preprocessing data...")
    processed_data, scaler = preprocess_data(config_path)
    X = processed_data[features]
    y = processed_data[target]

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    logging.info("Training Random Forest model...")
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    logging.info("Classification report...")
    print(classification_report(y_test, y_pred))

    # 保存模型
    model_path = "models/failure_cause_model.pkl"
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")
    return model

if __name__ == "__main__":
    train_model("config/config.yaml")
