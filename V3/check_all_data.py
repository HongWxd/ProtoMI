import numpy as np
from sklearn.decomposition import PCA
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_path = './data/all_data.pkl'
with open(data_path, 'rb') as f:
    dataset = pickle.load(f)

# node_counts = [data.num_nodes for data in dataset]
# edge_counts = [data.num_edges for data in dataset]

# # 计算方差与标准差
# node_var = np.var(node_counts)
# edge_var = np.var(edge_counts)

# print(f"节点数方差: {node_var:.2f}, 标准差: {np.std(node_counts):.2f}")
# print(f"边数方差: {edge_var:.2f}, 标准差: {np.std(edge_counts):.2f}")

# # 简单可视化
# plt.figure(figsize=(10,10))
# sns.jointplot(x=node_counts, y=edge_counts, kind="scatter", color="#4C72B0")
# plt.xlabel("Number of nodes")
# plt.ylabel("Number of edges")
# plt.title("Graph size distribution (nodes vs edges)", y=1.02)
# plt.tight_layout()
# plt.savefig("./V3/plots/graph_size_distribution.png", dpi=300)
# plt.clf()



# 将所有节点特征堆叠起来
all_node_features = []
for data in dataset:
    if hasattr(data, 'x') and data.x is not None:
        all_node_features.append(data.x.numpy())

X = np.vstack(all_node_features)

# PCA 分析
pca = PCA(n_components=min(10, X.shape[1]))  # 取前10维
pca.fit(X)

explained = np.cumsum(pca.explained_variance_ratio_)

loadings = pca.components_  # shape = [n_components, n_features]
n_components = loadings.shape[0]
n_features = loadings.shape[1]

# 取前几个主成分（例如前5个）
top_k = min(10, n_components)
loadings_subset = loadings[:top_k, :]

# 如果你有特征名称（如原子属性名），可以替换这里
feature_names = [f"F{i}" for i in range(n_features)]

# 可视化热力图
plt.figure(figsize=(12, 6))
sns.heatmap(loadings_subset,
            cmap="coolwarm",
            xticklabels=feature_names,
            yticklabels=[f"PC{i+1}" for i in range(top_k)],
            center=0,
            annot=False)

plt.title("Feature Contributions (Loadings) for Top Principal Components")
plt.xlabel("Original Node Features")
plt.ylabel("Principal Components")
plt.tight_layout()
plt.savefig("./V3/plots/pca_feature_loadings_heatmap.png", dpi=300)


# ==============================
# 2️⃣ PCA 投影散点图（前两主成分）
# ==============================
X_pca = pca.transform(X)

plt.figure(figsize=(6, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.6, color="#55A868")
plt.xlabel(f"PC1 ({explained[0]*100:.2f}% var)")
plt.ylabel(f"PC2 ({explained[1]*100:.2f}% var)")
plt.title("PCA Projection of Node Features")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("./V3/plots/pca_projection_scatter.png", dpi=300)

print("累计解释率（前10维）:")
for i, e in enumerate(explained):
    print(f"前 {i+1} 个主成分累计解释率: {e*100:.2f}%")

# # 可视化
# plt.figure(figsize=(6,4))
# plt.plot(np.arange(1, len(explained)+1), explained*100, marker='o', color="#C44E52")
# plt.title("PCA cumulative explained variance of node features")
# plt.xlabel("Number of principal components")
# plt.ylabel("Cumulative explained variance (%)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./V3/plots/pca_explained_variance.png", dpi=300)
