import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# load the smiles and fingerprint data then save them into .csv file
# path = '../data/raw/'
# files_path = os.listdir(path)
# files = [i for i in files_path if i.endswith('.xlsx')]
# smiles_list = []
# for file in tqdm(files):
#     data = pd.DataFrame(pd.read_excel(path + file, sheet_name='Substance Identifiers'))
#     df = data.iloc[4:].reset_index(drop=True)
#     substance_SMILES = df['Unnamed: 2'].values
#     for smile in substance_SMILES:
#         smiles_list.append(smile)
#
#
# new_df = pd.DataFrame()
# new_df['SMILES'] = smiles_list
# new_df.to_csv('./raw_smiles.csv', index=False)

# Load the saved data
# df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/searching_space_data.csv'))
df = pd.DataFrame(pd.read_csv('./data/labeled_data.csv'))
smiles_list = set(df['smile'].values.tolist())
rep_df = pd.DataFrame(pd.read_csv('./data/labeled_data.csv'))
rep_smiles_list = set(rep_df['smile'].values.tolist())

smiles = []
for smile in smiles_list:
    smile = str(smile)
    smiles.append(smile)

data_labels = []
for smile in smiles:
    label_list = rep_df.loc[rep_df['smile'] == smile, 'label'].values
    label_list = [str(i) for i in label_list]

    if '1' in label_list:
        data_labels.append(1)
    elif '1' not in label_list and '0' in label_list:
        data_labels.append(0)
    elif '1' not in label_list and '0' not in label_list and '-1' in label_list:
        data_labels.append(-1)

# create a list of mols
mols = []
none_smiles = []
for smile in smiles:
    if Chem.MolFromSmiles(smile) is None:
        none_smiles.append(smile)
    else:
        mols.append(Chem.MolFromSmiles(smile))
smiles = [i for i in smiles if i not in none_smiles]

# create a list of fingerprints from mols
fps = [Chem.RDKFingerprint(mol) for mol in tqdm(mols)]

# normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(fps)
# X_scaled = np.array(fps)

# use the clustering algorithm
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# # PCA linear
# pca = PCA(n_components=2)
# X_embedded = pca.fit_transform(X_scaled)

# t-NSE none-linear
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(X_scaled)

# for i in range(len(X_scaled)):
#     print(list(rep_smiles_list)[i], X_embedded[i, 0], X_embedded[i, 1])

# # 可视化降维后的数据
# plt.figure(figsize=(8, 6))
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', s=50)

# plt.title(f'KMeans Clustering Result (t-SNE {n_clusters} clusters)')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.savefig(f'./figs/{n_clusters}_clustering_result.jpg', dpi=600)


# # print("各主成分的方差解释比例:", pca.explained_variance_ratio_)
# plt.figure(figsize=(8, 6))
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='blue', label='PCA Transformed Data')
# plt.title("PCA of Molecular Fingerprints")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend()
# plt.show()

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制3D散点图
# ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='blue', label='PCA Transformed Data')
#
# # 设置标题和标签
# ax.set_title("3D PCA of Molecular Fingerprints")
# ax.set_xlabel("Principal Component 1")
# ax.set_ylabel("Principal Component 2")
# ax.set_zlabel("Principal Component 3")
# ax.legend()


plt.figure()
# colors = ["navy", "turquoise", "darkorange", "yellowgreen"]
colors = ["darkorange", "yellowgreen"]
lw = 2
# target_names = ['Li', 'Na', 'Li_Na', 'Other']
# target_names = ['SEI', 'no_SEI', 'Unreported']
# target_names = ['SEI', 'no_SEI']
target_names = ['SEI']

y = []
for smile, data_label in zip(tqdm(smiles), data_labels):
    # Analysis whole substances
    if data_label == 1:
        y.append(0)
    elif data_label == 0:
        y.append(1)
    else:
        y.append(2)

y = np.array(y)
markers = ['o', 's', 'x']
for color, i, target_name, marker in zip(colors, [0], target_names, markers):
    plt.scatter(
        X_embedded[y == i, 0], X_embedded[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name, marker=marker
    )

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("t-SNE of reported data")
plt.tight_layout()
plt.savefig('./figs/reported_t-SNE_SEI.jpg', dpi=600)
# plt.show()


