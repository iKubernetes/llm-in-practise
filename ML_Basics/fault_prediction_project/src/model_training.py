# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  # 导入梯度提升分类器
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold  # 导入训练集划分、随机搜索和交叉验证
from sklearn.preprocessing import StandardScaler  # 导入标准化处理
from sklearn.metrics import classification_report, roc_curve, auc  # 导入评估指标
from imblearn.over_sampling import SMOTE  # 导入 SMOTE 用于处理不平衡数据
from scipy.stats import uniform, randint  # 导入分布函数，用于超参数随机搜索
import joblib  # 用于模型和标准化器的保存
import matplotlib.pyplot as plt  # 导入可视化工具
from feature_engineering import extract_features  # 导入特征工程模块

# 加载原始数据
data = pd.read_csv("data/raw/system_metrics.csv")

# 使用特征工程提取特征
data, features = extract_features(data)

# 分离特征和标签，填充缺失值为 0
X = data[features].fillna(0)
y = data['label']

# 标准化特征数据，确保每个特征的均值为 0，方差为 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 SMOTE 方法对不平衡数据进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 划分训练集和测试集，80% 用于训练，20% 用于测试
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# 定义超参数的分布范围，用于随机搜索
param_dist = {
    'n_estimators': randint(50, 200),  # 弱学习器的数量
    'learning_rate': uniform(0.01, 0.2),  # 学习率
    'max_depth': randint(3, 8),  # 每个树的最大深度
    'min_samples_split': randint(2, 10),  # 内部节点再划分所需最小样本数
    'min_samples_leaf': randint(1, 5)  # 叶子节点最小样本数
}

# 设置分层交叉验证，确保每个折叠中各类样本比例相同
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 设置随机搜索，选择最佳的超参数组合
model = GradientBoostingClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=model,  # 选择模型
    param_distributions=param_dist,  # 超参数分布
    n_iter=20,  # 随机搜索的次数
    cv=cv,  # 使用交叉验证
    scoring='recall',  # 以召回率为标准评估模型
    n_jobs=-1,  # 使用所有可用的 CPU 核心
    verbose=1,  # 输出详细过程
    random_state=42
)

# 进行随机搜索，训练模型
random_search.fit(X_train, y_train)

# 输出最佳参数和最佳交叉验证召回率
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Recall:", random_search.best_score_)

# 使用最佳模型对测试集进行预测
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# 输出测试集的分类报告
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred))

# 输出特征重要性，按重要性排序
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# 保存最佳模型和标准化器
joblib.dump(best_model, "models/gbc_fault_prediction_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# 可视化 ROC 曲线
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 获取预测为正类的概率
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # 计算假阳性率和真阳性率
roc_auc = auc(fpr, tpr)  # 计算 AUC（曲线下面积）

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

plt.figure()
plt.plot(fpr, tpr, label=f'ROC 曲线 (AUC = {roc_auc:.2f})')  # 绘制 ROC 曲线
plt.plot([0, 1], [0, 1], 'k--')  # 绘制随机分类器的对角线
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC 曲线')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")  # 保存 ROC 曲线图像
plt.close()  # 关闭图形
