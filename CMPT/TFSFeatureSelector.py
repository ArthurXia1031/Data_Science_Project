import pandas as pd
import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class TopologicalFeatureSelector(BaseEstimator, TransformerMixin):
    """
    基于拓扑图论的特征筛选器 (TFS - Simplified via MST)
    逻辑:
    1. 计算特征间的 Spearman 相关性矩阵。
    2. 构建完全图，边权重为相关性的绝对值。
    3. 生成最大生成树 (Maximum Spanning Tree, MST) 以过滤噪声和冗余边。
    4. 计算 MST 中各节点的度中心性 (Degree Centrality)。
    5. 保留中心性最高的 Top K 个特征。
    """
    
    def __init__(self, n_features_to_select=50, method='spearman'):
        self.n_features_to_select = n_features_to_select
        self.method = method
        self.selected_features_ = []
        self.feature_scores_ = {}
        self.G_ = None # 保存图结构用于后续分析/可视化
        
    def fit(self, X, y=None):
        """
        X: pandas DataFrame, 包含所有 V_A 原始特征
        """
        # 1. 数据预处理 (标准化) - 虽然Spearman对缩放不敏感，但为了稳健性建议保留
        # 注意：为了计算相关性，我们直接用原始数据的 Rank 即可，这里假设 X 已经是 DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
            
        print(f"Step 1: Calculating {self.method} correlation matrix for {X.shape[1]} features...")
        # 金融数据通常是非线性的，Spearman (Rank Correlation) 比 Pearson 更稳健
        corr_matrix = X.corr(method=self.method).abs()
        
        # 2. 构建图结构 (完全图)
        print("Step 2: Building graph structure...")
        G = nx.Graph()
        features = X.columns.tolist()
        G.add_nodes_from(features)
        
        # 为了利用 NetworkX 的 MST 算法，我们需要将边列表化
        # 这是一个 O(N^2) 的操作，对于 227 个特征非常快
        edges = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                u, v = features[i], features[j]
                weight = corr_matrix.iloc[i, j]
                # NetworkX 的 maximum_spanning_tree 需要权重
                edges.append((u, v, weight))
        
        G.add_weighted_edges_from(edges)
        
        # 3. 拓扑过滤 (Topological Filtering) - 提取骨干网络
        # 使用最大生成树 (MST) 替代 TMFG (MST 是 TMFG 的子集，足以捕捉核心结构)
        print("Step 3: Extracting Maximum Spanning Tree (MST) backbone...")
        # 这一步会自动切断那些由"多重共线性"引起的冗余弱连接
        mst_G = nx.maximum_spanning_tree(G, weight='weight')
        self.G_ = mst_G
        
        # 4. 计算中心性 (Centrality Calculation)
        print("Step 4: Calculating Degree Centrality...")
        # 在 MST 中，连接度高的节点就是信息的“枢纽”
        centrality = nx.degree_centrality(mst_G)
        
        # 排序并选择 Top K
        sorted_features = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        self.selected_features_ = [feat for feat, score in sorted_features[:self.n_features_to_select]]
        self.feature_scores_ = dict(sorted_features)
        
        print(f"Done. Selected {len(self.selected_features_)} features.")
        return self

    def transform(self, X):
        """
        返回筛选后的 DataFrame
        """
        if not self.selected_features_:
            raise ValueError("Selector has not been fitted yet.")
        return X[self.selected_features_]

# --- Demo: 如何在你的项目中使用 ---

# 1. 模拟生成 Discover 的 V_A 数据 (假设有 227 个特征，存在高度共线性)
def generate_dummy_data(n_samples=1000, n_features=227):
    np.random.seed(42)
    # 生成一些基础特征
    base_data = np.random.randn(n_samples, 50)
    # 生成大量高度相关的冗余特征 (模拟 Monthly_0, Monthly_1 等)
    redundant_data = base_data[:, :10] * 0.9 + np.random.randn(n_samples, 10) * 0.1
    # 拼凑成 DataFrame
    cols = [f'V_A_feat_{i}' for i in range(n_features)]
    data = np.random.randn(n_samples, n_features)
    # 注入相关性
    data[:, :10] = redundant_data # 前10个特征高度相关
    return pd.DataFrame(data, columns=cols)

# 假设这是你从 Spark 导出的 Sample Data (Pandas DataFrame)
df_va_features = generate_dummy_data()

# 2. 实例化 TFS 选择器
tfs = TopologicalFeatureSelector(n_features_to_select=50, method='spearman')

# 3. 训练并筛选
tfs.fit(df_va_features)

# 4. 获取结果
selected_df = tfs.transform(df_va_features)
print("\nTop 10 Selected Features based on Centrality:")
print(tfs.selected_features_[:10])

# 5. (可选) 可视化特征网络 - 看看谁是 Hub
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(tfs.G_, k=0.15)
# nx.draw(tfs.G_, pos, with_labels=False, node_size=10, alpha=0.5)
# # 绘制选中的节点为红色
# nx.draw_networkx_nodes(tfs.G_, pos, nodelist=tfs.selected_features_, node_color='r', node_size=30)
# plt.title("Feature Topology (MST) - Red Nodes are Selected")
# plt.show()
