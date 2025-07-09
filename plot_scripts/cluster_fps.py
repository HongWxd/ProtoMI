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
import umap
import seaborn as sns
warnings.filterwarnings('ignore')


df = pd.DataFrame(pd.read_csv('./data/predict_1.csv'))
cid_list = set(df['cid'].values.tolist())
pred_len = len(cid_list)
label_df = pd.DataFrame(pd.read_csv('./data/labeled_data.csv'))
label_cid_list = set(label_df['cid'].values.tolist()) 
cid_list = np.concatenate((list(cid_list), list(label_cid_list)), axis=0)
searching_space_df = pd.DataFrame(pd.read_csv('./data/searching_space_data.csv'))

formulas = []
smiles = []
for cid in cid_list:
    formula = searching_space_df.loc[searching_space_df['cid'] == float(cid), 'formula'].values[0]
    smile = searching_space_df.loc[searching_space_df['cid'] == float(cid), 'SMILES'].values[0]
    formulas.append(formula)
    smiles.append(smile)

# create a list of mols
mols = []
none_smiles = []
for smile in smiles:
    if Chem.MolFromSmiles(smile) is None:
        none_smiles.append(smile)
    else:
        mols.append(Chem.MolFromSmiles(smile))

# create a list of fingerprints from mols
fps = [Chem.RDKFingerprint(mol) for mol in tqdm(mols)]
print(len(fps))

# normalization
n_clusters = 10
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(fps)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
labels_kmeans = labels_kmeans[:pred_len]

reducer_2d = umap.UMAP(random_state=42)
emb_2d = reducer_2d.fit_transform(X_scaled)
reducer_3d = umap.UMAP(n_components=3, random_state=42)
emb_3d = reducer_3d.fit_transform(X_scaled)

pred_emb_2d = emb_2d[:pred_len]
labeled_emb_2d = emb_2d[pred_len:]
pred_emb_3d = emb_3d[:pred_len]
labeled_emb_3d = emb_3d[pred_len:]

fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111, projection='3d')
colors = sns.color_palette("husl", n_clusters)
marker_list = [
        'o', 's', '^', 'v', '<', '>', 'D', 'd', 'p', 'h',
        'H', '*', '+', 'x', 'X', '|', '_', '.', ',', 'P'
    ]

# plot predicted points
for cluster_id in range(n_clusters):
    idx = labels_kmeans == cluster_id
    plt.scatter(pred_emb_2d[idx, 0], pred_emb_2d[idx, 1], 
                color=colors[cluster_id], 
                marker=marker_list[cluster_id % len(marker_list)],
                label=f'Cluster {cluster_id + 1}', 
                s=45)
    # ax.scatter(pred_emb_3d[idx, 0], pred_emb_3d[idx, 1], pred_emb_3d[idx, 2], 
    #             color=colors[cluster_id], 
    #             marker=marker_list[cluster_id % len(marker_list)],
    #             label=f'Cluster {cluster_id + 1}', 
    #             s=45)
    
# plot labeled points
# ax.scatter(labeled_emb_3d[:, 0], labeled_emb_3d[:, 1], labeled_emb_3d[:, 2], marker='2', color='red', label='Real label', s=45)
plt.scatter(labeled_emb_2d[:,0], labeled_emb_2d[:,1], marker='2', color='red', label='Real label', s=45)
plt.legend(title='Cluster Labels', bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
plt.title('UMAP projection with KMeans clusters')
plt.tight_layout()
plt.grid(True)
plt.savefig('./figs/KMeans_clusters_fps.png', dpi=600)

df = pd.DataFrame()
df['cid'] = cid_list[:pred_len].tolist()
df['formula'] = formulas[:pred_len]
df['smiles'] = smiles[:pred_len]
df['x'] = pred_emb_2d[:, 0]
df['y'] = pred_emb_2d[:, 1]
df['cluster'] = [i + 1 for i in labels_kmeans.tolist()]

labeled_df = pd.DataFrame()
labeled_df['cid'] = cid_list[pred_len:].tolist()
labeled_df['formula'] = formulas[pred_len:]
labeled_df['smiles'] = smiles[pred_len:]
labeled_df['x'] = labeled_emb_2d[:, 0]
labeled_df['y'] = labeled_emb_2d[:, 1]

df.to_csv('./data/kmeans_clusters.csv', index=False)
labeled_df.to_csv('./data/labeled_UMAP.csv', index=False)