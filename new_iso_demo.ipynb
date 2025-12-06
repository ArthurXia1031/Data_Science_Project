import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import warnings

# 忽略一些无关紧要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置区域 (Configuration)
# ==========================================
# 在这里指定你的文件名和目标列名
DATA_FILE_PATH = 'your_data.csv'  # 替换为你的真实csv文件名
TARGET_COL = 'fraud_ind'          # 你的label列名 (0/1)
ID_COLS = ['transaction_id', 'user_id', 'date'] # 如果你明确知道哪些是ID，填在这里；否则脚本会自动尝试识别

# ==========================================
# 2. 智能数据清洗函数 (Auto Data Cleaning)
# ==========================================
def clean_and_preprocess(df, target_col, exclude_cols=[]):
    """
    自动处理脏数据：剔除ID、填充缺失值、编码分类变量
    """
    print(">>> 开始数据清洗...")
    
    # 1. 分离特征和标签
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        raise ValueError(f"数据中未找到目标列 '{target_col}'")

    # 2. 剔除明确指定的无关列
    cols_to_drop = [c for c in exclude_cols if c in X.columns]
    if cols_to_drop:
        print(f"剔除指定列: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)

    # 3. 自动识别并剔除 ID 类特征 (Unique 值过多的 Object 列)
    # 阈值：如果某列 90% 的值都是唯一的，且是字符串，通常是 ID
    for col in X.select_dtypes(include=['object', 'string']).columns:
        unique_ratio = X[col].nunique() / len(X)
        if unique_ratio > 0.9:
            print(f"剔除高基数特征 (疑似ID): {col} (Unique Ratio: {unique_ratio:.2f})")
            X = X.drop(columns=[col])
        elif X[col].nunique() == 1:
            print(f"剔除单一值特征 (无方差): {col}")
            X = X.drop(columns=[col])

    # 4. 缺失值填充 (简单的策略：数值填中位数，分类填 'Unknown')
    # 实际生产中可能需要更精细的 Imputation
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna('Unknown')

    # 5. 分类特征编码 (Label Encoding)
    # 对于 iForest 和 树模型，Label Encoding 通常够用且比 One-Hot 更省内存
    print("正在编码分类特征...")
    for col in cat_cols:
        le = LabelEncoder()
        # 强制转换为字符串以处理混合类型
        X[col] = le.fit_transform(X[col].astype(str))
        
    print(f"清洗完成。特征维度: {X.shape}")
    return X, y

# ==========================================
# 3. 主流程
# ==========================================

try:
    # 1. 加载数据
    print(f"正在读取数据: {DATA_FILE_PATH} ...")
    # 既然数据有25万行，正常读取即可。如果内存紧张，可以加 low_memory=False
    # df = pd.read_csv(DATA_FILE_PATH)
    
    # --- 为了演示代码运行，这里依然保留生成数据的逻辑，实际使用时请注释掉下面这行，取消上面 read_csv 的注释 ---
    # 生成 25万行数据
    def generate_mock_data():
        np.random.seed(42)
        n_normal = 245099
        n_fraud = 4671
        X_normal = pd.DataFrame({
            'trans_id': range(n_normal), # 脏数据：ID列
            'amt': np.random.normal(100, 20, n_normal),
            'mcc': np.random.choice(['5411', '5812', '5999'], n_normal), # 脏数据：分类列
            'velocity_1h': np.random.poisson(2, n_normal),
            'null_col': [np.nan]*n_normal # 脏数据：全空列
        })
        X_fraud = pd.DataFrame({
            'trans_id': range(n_normal, n_normal+n_fraud),
            'amt': np.random.normal(500, 100, n_fraud),
            'mcc': np.random.choice(['5411', '5812', '5999'], n_fraud),
            'velocity_1h': np.random.normal(15, 5, n_fraud),
            'null_col': [np.nan]*n_fraud
        })
        X = pd.concat([X_normal, X_fraud]).reset_index(drop=True)
        y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        X[TARGET_COL] = y
        return X
        
    df = generate_mock_data()
    # -----------------------------------------------------------------------------------------

    # 2. 数据清洗
    X, y = clean_and_preprocess(df, TARGET_COL, exclude_cols=ID_COLS)

    # 3. 训练 Isolation Forest (使用全量数据)
    # 25万行对 sklearn 来说是小菜一碟，不需要降采样
    real_fraud_rate = y.mean()
    contamination_rate = min(real_fraud_rate * 3, 0.1) # 稍微放宽一点阈值
    
    print(f"\n[Step 1] 训练 Isolation Forest (全量数据 {len(X)} 行)...")
    t0 = time.time()
    # n_jobs=-1 并行计算，max_samples='auto' 会自动抽样每棵树的样本数(通常256)，保证速度极快
    clf = IsolationForest(n_estimators=100, 
                          max_samples='auto', 
                          contamination=contamination_rate, 
                          random_state=42, 
                          n_jobs=-1)
    clf.fit(X)
    print(f"训练完成，耗时: {time.time()-t0:.2f} 秒")

    # 4. 计算异常分
    scores = -clf.decision_function(X) # 分数越高越异常
    threshold = np.percentile(scores, 100 * (1 - contamination_rate))
    
    # 5. 锁定“真异常”群体 (True Anomalies)
    # 我们只关心：模型觉得异常 且 真实标签也是Fraud 的部分
    high_risk_indices = np.where((scores > threshold) & (y == 1))[0]
    
    print(f"\n[Step 2] 结果分析")
    print(f"异常分阈值: {threshold:.4f}")
    print(f"捕获到的高分欺诈样本数: {len(high_risk_indices)}")
    
    if len(high_risk_indices) == 0:
        print("警告：模型未在高分段捕获到欺诈样本。可能需要调整特征工程。")
    else:
        # 6. SHAP 解释 (关键性能优化点)
        # 即使有25万数据，我们只需要解释那 几百个 最像异常的欺诈样本
        # 不要对全量 X 计算 SHAP，那是算不完的
        
        # 采样策略：最多取 500 个样本做解释，足够发现规律了
        sample_size = min(500, len(high_risk_indices))
        explain_indices = np.random.choice(high_risk_indices, sample_size, replace=False)
        X_explain = X.iloc[explain_indices]
        
        print(f"\n[Step 3] SHAP 归因分析 (采样 {sample_size} 个样本)...")
        t1 = time.time()
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_explain)
        print(f"SHAP 计算完成，耗时: {time.time()-t1:.2f} 秒")
        
        # 打印特征重要性
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values(by='importance', ascending=False)
        
        print("\n>>> 对异常判定贡献最大的特征:")
        print(feature_importance.head(5))

        # 7. 规则提取
        print(f"\n[Step 4] 生成业务规则...")
        # 正样本：被抓出的高分欺诈
        # 负样本：普通的正常交易 (1:1 采样)
        X_pos = X.iloc[high_risk_indices]
        y_pos = np.ones(len(X_pos))
        
        normal_indices = np.where(scores < threshold)[0]
        sample_neg_indices = np.random.choice(normal_indices, len(X_pos), replace=False)
        X_neg = X.iloc[sample_neg_indices]
        y_neg = np.zeros(len(X_neg))
        
        X_rule = pd.concat([X_pos, X_neg])
        y_rule = np.concatenate([y_pos, y_neg])
        
        # 决策树提取规则
        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)
        dt.fit(X_rule, y_rule)
        
        print("\n" + "="*50)
        print(">>> 推荐的高精度规则逻辑 (可直接用于规则引擎) <<<")
        print("="*50)
        print(export_text(dt, feature_names=list(X.columns)))

except Exception as e:
    print(f"发生错误: {e}")
    print("建议检查: 1.文件名是否正确 2.内存是否溢出(尝试减少列数)")
