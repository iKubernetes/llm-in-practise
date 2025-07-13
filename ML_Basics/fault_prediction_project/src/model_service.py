# src/model_service.py
from flask import Flask, request  # 导入 Flask 框架和请求模块
import joblib  # 导入 joblib 用于加载保存的模型
import pandas as pd  # 导入 pandas 用于数据处理
import numpy as np  # 导入 numpy 用于数值计算
import os  # 导入 os 用于获取环境变量
from feature_engineering import extract_features  # 导入特征工程模块中的 extract_features 函数

# 初始化 Flask 应用
app = Flask(__name__)

# 加载训练好的模型和标准化器
model = joblib.load("models/gbc_fault_prediction_model.pkl")  # 加载故障预测模型
scaler = joblib.load("models/scaler.pkl")  # 加载标准化器

# 健康检查路由，用于确认服务是否正常运行
@app.route("/health")
def health():
    return {"status": "healthy"}  # 返回健康状态

# 故障预测路由，使用 POST 方法接收请求数据
@app.route("/predict_fault", methods=["POST"])
def predict_fault():
    try:
        # 获取请求中的 JSON 数据
        data = request.json
        
        # 将接收到的数据转化为 DataFrame 格式，列名与训练时的特征一致
        df = pd.DataFrame([data], columns=['cpu_usage', 'ram_usage', 'disk_io', 'temperature', 'error_count', 'timestamp', 'device_id'])
        
        # 提取特征
        df, features = extract_features(df)
        
        # 填充缺失值为 0，并进行标准化
        X = df[features].fillna(0)
        X_scaled = scaler.transform(X)
        
        # 使用加载的模型进行预测
        prediction = model.predict(X_scaled)  # 预测故障（0或1）
        probability = model.predict_proba(X_scaled)[:, 1]  # 获取预测为故障的概率
        
        # 返回预测结果和特征重要性
        return {
            "is_fault_predicted": bool(prediction[0] == 1),  # 如果预测结果为 1，则为故障
            "fault_probability": float(probability[0]),  # 预测为故障的概率
            "feature_importance": dict(zip(features, model.feature_importances_))  # 返回特征重要性
        }
    
    except Exception as e:
        # 异常处理，返回错误信息
        return {"error": str(e)}, 500

# 启动 Flask 应用
if __name__ == "__main__":
    # 获取环境变量中的端口号，默认为 5000
    port = int(os.getenv("PORT", 5000))
    
    # 启动服务，监听所有 IP 地址的请求
    app.run(host="0.0.0.0", port=port)
