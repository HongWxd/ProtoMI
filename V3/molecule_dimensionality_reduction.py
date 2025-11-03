from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from rdkit import DataStructs

def tanimoto_matrix(X):
    intersection = X @ X.T
    bit_sum = X.sum(axis=1)
    union = bit_sum[:, None] + bit_sum[None, :] - intersection
    union = np.where(union == 0, 1, union)
    sim = intersection / union
    return sim


searching_space_df = pd.DataFrame(pd.read_csv("./data/searching_space_data_V2.csv"))
searching_space_smiles = searching_space_df['SMILES'].tolist()
with open('./V3/processed_data/additives.json', "r", encoding="utf-8") as f:
    additives_data = json.load(f)
additives_smiles = [i['smiles'] for i in additives_data.values()]
additives_names = [i for i in additives_data.keys()]

smiles_list = additives_smiles

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
fps = []
valid_smiles = []
labels = []
orgin_fps = []
for smi in tqdm(smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = morgan_gen.GetFingerprint(mol)  
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        orgin_fps.append(fp)
        valid_smiles.append(smi)

        if smi in additives_smiles:
            labels.append("additive")
        else:
            labels.append("searching_space")

fps = np.array(fps)
X = fps.astype(int)
N, F = X.shape
bit_counts = X.sum(axis=1)        # 每个分子的 1 的数量
avg_bits = bit_counts.mean()
median_bits = np.median(bit_counts)
sparsity = 1.0 - (bit_counts.sum() / (N * F))

print(f"N={N}, F={F}")
print(f"平均 set bits per fingerprint = {avg_bits:.2f}, 中位数 = {median_bits}")
print(f"总体稀疏度 (fraction zeros) = {sparsity:.4f}")

# 1️⃣ 计算距离矩阵（建议用 cosine 或 euclidean）
# dist_matrix = pairwise_distances(fps, metric='euclidean')

sim_matrix = tanimoto_matrix(fps)
# 转化为距离矩阵用于降维
dist_matrix = 1 - sim_matrix
np.fill_diagonal(dist_matrix, 0)  # 对角线必须为 0


reducer_2d = umap.UMAP(random_state=42)
embeddings = reducer_2d.fit_transform(dist_matrix)
# reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='precomputed', random_state=42)
# embeddings = reducer_3d.fit_transform(dist_matrix)

n_add = len(additives_smiles)
additive_emb = embeddings[:n_add]
search_emb = embeddings[n_add:]


# 2️⃣ 进行层次聚类 (ward/linkage 可换成 'average'、'complete')
Z = linkage(dist_matrix, method='ward')

# 3️⃣ 画树状图
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=additives_names, leaf_rotation=90)
plt.title("Hierarchical Clustering of Additives (126 positive samples)")
plt.xlabel("Additive")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("./V3/plots/additives_hierarchical_clustering.png", dpi=600)


possible_clusters = range(2, 11)  # 尝试从2到10个簇
best_score = -1
best_k = None
best_labels = None

for k in possible_clusters:
    cluster_labels = fcluster(Z, t=k, criterion='maxclust')
    try:
        score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
        print(f"k={k}, silhouette score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = cluster_labels 
    except Exception as e:
        print(f"k={k} failed: {e}")

print(f"\n✅ 最佳簇数: {best_k}, 对应的平均轮廓系数: {best_score:.4f}")

# 保存最佳聚类结果
additive_df = pd.DataFrame({
    "Name": additives_names,
    "Cluster": best_labels
})
additive_df.to_csv("./V3/processed_data/additive_clustered_best.csv", index=False)
print("✅ 已保存: additive_clustered_best.csv")

# 可视化不同簇在UMAP上的分布
plt.figure(figsize=(8,6))
for i in range(1, best_k+1):
    plt.scatter(additive_emb[best_labels==i, 0], additive_emb[best_labels==i, 1], s=40, label=f"Cluster {i}", alpha=0.7)
plt.legend()
plt.title(f"UMAP Projection (Best Clusters = {best_k})")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("./V3/plots/additives_umap_best_cluster.png", dpi=600)

# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection='3d')

# for i in range(1, best_k+1):
#     idx = best_labels == i
#     ax.scatter(
#         additive_emb[idx,0], additive_emb[idx,1], additive_emb[idx,2],
#         s=40,
#         label=f"Cluster {i}",
#         alpha=0.7
#     )

# ax.set_xlabel("UMAP-1")
# ax.set_ylabel("UMAP-2")
# ax.set_zlabel("UMAP-3")
# ax.set_title(f"3D UMAP Projection (Best Clusters = {best_k})")
# ax.legend()
# plt.tight_layout()
# plt.savefig("./V3/plots/additives_umap_3d_best_cluster.png", dpi=600)