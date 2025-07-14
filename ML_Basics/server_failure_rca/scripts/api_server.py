import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import List
from datetime import datetime

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_config, setup_logging

# 初始化日志
setup_logging("logs/api_server.log", "INFO")
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI(title="Server Failure Root Cause Analysis API", 
              description="API for predicting server failure causes using a pre-trained model",
              version="1.0.0")

# 定义输入数据模型
class MetricInput(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU使用率（%）")
    memory_usage: float = Field(..., ge=0, le=100, description="内存使用率（%）")
    disk_io: float = Field(..., ge=0, description="磁盘I/O（MB/s）")
    network_latency: float = Field(..., ge=0, description="网络延迟（ms）")

# 定义批量输入模型
class BatchMetricInput(BaseModel):
    metrics: List[MetricInput]

# 全局变量存储模型和配置
MODEL = None
SCALER = None
FEATURES = None

@app.on_event("startup")
async def startup_event():
    """在应用启动时加载模型和配置"""
    global MODEL, SCALER, FEATURES
    try:
        config = load_config("config/config.yaml")
        model_path = "models/failure_cause_model.pkl"
        scaler_path = "models/scaler.pkl"
        FEATURES = config['data']['features']
        
        # 加载模型
        logger.info(f"加载模型从 {model_path}")
        MODEL = joblib.load(model_path)
        
        # 加载标准化器
        if not os.path.exists(scaler_path):
            logger.error(f"标准化器文件 {scaler_path} 不存在")
            raise HTTPException(status_code=500, detail="标准化器文件不存在，请确保模型训练已完成")
        logger.info(f"加载标准化器从 {scaler_path}")
        SCALER = joblib.load(scaler_path)
        
        logger.info("模型和标准化器加载成功")
    except Exception as e:
        logger.error(f"启动时加载模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器初始化失败: {str(e)}")

@app.post("/predict", response_model=dict)
async def predict_failure_cause(metric: MetricInput):
    """预测单个数据点的故障原因"""
    try:
        # 转换为DataFrame
        data = pd.DataFrame([metric.dict()], columns=FEATURES)
        
        # 标准化输入
        data_scaled = SCALER.transform(data)
        
        # 预测
        prediction = MODEL.predict(data_scaled)[0]
        probabilities = MODEL.predict_proba(data_scaled)[0]
        prob_dict = {cls: round(prob, 4) for cls, prob in zip(MODEL.classes_, probabilities)}
        
        logger.info(f"预测完成: 输入={metric.dict()}, 结果={prediction}")
        return {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "prediction": prediction,
            "probabilities": prob_dict
        }
    except Exception as e:
        logger.error(f"预测错误: {str(e)}")
        raise HTTPException(status_code=400, detail=f"预测失败: {str(e)}")

@app.post("/batch_predict", response_model=dict)
async def batch_predict_failure_cause(batch: BatchMetricInput):
    """预测批量数据点的故障原因"""
    try:
        # 转换为DataFrame
        data = pd.DataFrame([m.dict() for m in batch.metrics], columns=FEATURES)
        
        # 标准化输入
        data_scaled = SCALER.transform(data)
        
        # 预测
        predictions = MODEL.predict(data_scaled)
        probabilities = MODEL.predict_proba(data_scaled)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            prob_dict = {cls: round(prob, 4) for cls, prob in zip(MODEL.classes_, probs)}
            results.append({
                "index": i,
                "prediction": pred,
                "probabilities": prob_dict
            })
        
        logger.info(f"批量预测完成: 样本数={len(batch.metrics)}")
        return {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "predictions": results
        }
    except Exception as e:
        logger.error(f"批量预测错误: {str(e)}")
        raise HTTPException(status_code=400, detail=f"批量预测失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
