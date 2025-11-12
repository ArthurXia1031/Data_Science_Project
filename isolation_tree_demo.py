# ======================================
#  Isolation Forest 单账户级异常检测模板
# ======================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# ---- 1. 数据加载 -------------------------------------------------
# df: 单账户数据，每行是一笔交易，包含 fraud_ind(0/1) 和 300+ features
# 例: df = pd.read_csv("account_1234_txn.csv")

y = df['fraud_ind']
X = df.drop(columns=['fraud_ind'])

# ---- 2. 标准化 ---------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- 3. 训练 IsolationForest ------------------------------------
iso = IsolationForest(
    n_estimators=300,
    contamination=0.05,     # 调整为 fraud 占比的 2~3 倍更稳健
    max_samples='auto',
    random_state=42
)
df['iso_pred'] = iso.fit_predict(X_scaled)
df['iso_score'] = -iso.decision_function(X_scaled)  # 数值越大越异常

# ---- 4. 可视化 & 验证 ------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x=y, y=df['iso_score'])
plt.title("Anomaly Score by Fraud Indicator")
plt.show()

auc = roc_auc_score(y, df['iso_score'])
ap = average_precision_score(y, df['iso_score'])
print(f"AUC={auc:.3f},  AP={ap:.3f}")

# ---- 5. SHAP 解释（特征贡献） -----------------------------------
explainer = shap.Explainer(iso, X_scaled)
shap_values = explainer(X_scaled)

# 全局特征重要性
shap.summary_plot(shap_values, X, plot_type='bar')

# 对最异常的交易展示局部解释
idx = df['iso_score'].idxmax()
print(f"\n最异常交易 index={idx}, fraud_ind={y.iloc[idx]}")
shap.plots.waterfall(shap_values[idx])

# ---- 6. 导出可供 case review 的表 -------------------------------
cols_out = ['iso_score'] + list(X.columns) + ['fraud_ind']
df_out = df[cols_out].sort_values('iso_score', ascending=False)
df_out.head(20)
