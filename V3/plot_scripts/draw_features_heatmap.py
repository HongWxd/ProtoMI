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
        all_node_features.append(data.edge_attr.numpy())

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
print(loadings_subset.shape)

# 如果你有特征名称（如原子属性名），可以替换这里
feature_names = ['bond_type_single', 'bond_type_double', 'bond_type_triple', 'bond_type_aromatic', 'bond_is_conjugated', 'bond_is_in_ring', 'bond_dir_none', 'bond_dir_beginwedge', 'bond_dir_begindash', 'bond_dir_enddownright', 'bond_dir_endupright', 'bond_is_aromatic', 'stereo_type_stereoz', 'stereo_type_stereoe', 'stereo_type_stereoany', 'stereo_type_stereonone']
# feature_names = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown', 'n_heavy_neighbors_0', 'n_heavy_neighbors_1', 'n_heavy_neighbors_2', 'n_heavy_neighbors_3', 'n_heavy_neighbors_4', 'n_heavy_neighbors_MoreThanFour', 'formal_charge_-3', 'formal_charge_-2', 'formal_charge_-1', 'formal_charge_0', 'formal_charge_1', 'formal_charge_2', 'formal_charge_3', 'formal_charge_Extreme', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER', 'is_in_a_ring', 'is_aromatic', 'atomic_mass_scaled', 'vdw_radius_scaled', 'covalent_radius_scaled', 'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER', 'n_hydrogens_0', 'n_hydrogens_1', 'n_hydrogens_2', 'n_hydrogens_3', 'n_hydrogens_4', 'n_hydrogens_MoreThanFour']

# low_contrib_mask = np.all(np.abs(loadings_subset) < 0.05, axis=0)

# # 保留剩下的列
# filtered_loadings = loadings_subset[:, ~low_contrib_mask]  # shape = [top_k, 剩余特征数]
# filtered_feature_names = [feature_names[i] for i, keep in enumerate(~low_contrib_mask) if keep]

# print("保留的特征数量:", len(filtered_feature_names))
# print("保留的特征名称:", filtered_feature_names)

# 可视化热力图
num_pc, num_feat = loadings_subset.shape

# 网格坐标
x, y = np.meshgrid(np.arange(num_feat), np.arange(num_pc))
x = x.flatten()
y = y.flatten()

values = loadings_subset.flatten()

# 气泡大小：用绝对值
sizes = np.abs(values) / np.max(np.abs(values)) * 300  

# 气泡颜色：用原始正负值
colors = values

plt.figure(figsize=(max(10, num_feat * 0.32), max(6, num_pc * 0.68)))

plt.scatter(
    x, y, 
    s=sizes, 
    c=colors, 
    cmap='coolwarm',   # 与原来保持一致
    edgecolors='grey',
    linewidth=0.3
)

plt.gca().invert_yaxis()  # PC1 在顶部
cbar = plt.colorbar(label='Loading Value')
cbar.set_label('Loading Value', fontsize=12)

plt.xticks(range(num_feat), feature_names, rotation=90, fontsize=14)
plt.yticks(range(num_pc), [f"PC{i+1}" for i in range(num_pc)], fontsize=14)

plt.title("PCA Feature Loadings - Bubble Heatmap", fontsize=18)
plt.tight_layout()
plt.savefig("./V3/plots/pca_feature_loadings_bubble_heatmap.png", dpi=600)


# # ==============================
# # 2️⃣ PCA 投影散点图（前两主成分）
# # ==============================
# X_pca = pca.transform(X)

# plt.figure(figsize=(6, 5))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.6, color="#55A868")
# plt.xlabel(f"PC1 ({explained[0]*100:.2f}% var)")
# plt.ylabel(f"PC2 ({explained[1]*100:.2f}% var)")
# plt.title("PCA Projection of edge Features")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.savefig("./V3/plots/pca_projection_scatter.png", dpi=300)

# # print("累计解释率（前10维）:")
# for i, e in enumerate(explained):
#     print(f"Rank {i+1} cumulative explanation rate of principal components: {e*100:.2f}%")

# # 可视化
# plt.figure(figsize=(6,4))
# plt.plot(np.arange(1, len(explained)+1), explained*100, marker='o', color="#C44E52")
# plt.title("PCA cumulative explained variance of edge features")
# plt.xlabel("Number of principal components")
# plt.ylabel("Cumulative explained variance (%)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("./V3/plots/pca_explained_variance.png", dpi=300)
