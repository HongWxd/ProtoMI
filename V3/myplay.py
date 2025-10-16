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

searching_space_df = pd.DataFrame(pd.read_csv("./data/searching_space_data.csv"))
searching_space_smiles = searching_space_df['SMILES'].tolist()
with open('./V3/processed_data/additives.json', "r", encoding="utf-8") as f:
    additives_data = json.load(f)
additives_smiles = [i['smiles'] for i in additives_data.values()]

smiles_list = additives_smiles + searching_space_smiles

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
fps = []
valid_smiles = []
labels = []

for smi in tqdm(smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = morgan_gen.GetFingerprint(mol)  
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        valid_smiles.append(smi)

        if smi in additives_smiles:
            labels.append("additive")
        else:
            labels.append("searching_space")

fps = np.array(fps)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(fps)
reducer_2d = umap.UMAP(random_state=42)
embeddings = reducer_2d.fit_transform(X_scaled)

n_add = len(additives_smiles)
additive_emb = embeddings[:n_add]
# search_emb = embeddings[n_add:]

print(len(additive_emb))

additive_df = pd.DataFrame(additive_emb, columns=['UMAP1', 'UMAP2'])
additive_df['additives'] = list(additives_data.keys())
additive_df['SMILES'] = additives_smiles
# additive_df.to_csv('./V3/processed_data/additive_embeddings.csv', index=False)

plt.figure(figsize=(8,7))
# plt.scatter(search_emb[:,0], search_emb[:,1], c='lightgray', s=20, label='Searching space')
plt.scatter(additive_emb[:,0], additive_emb[:,1], c='red', s=50, edgecolor='black', label='Additives')
plt.title("Molecular distribution", fontsize=13)
plt.xlabel("UMAP dimension 1")
plt.ylabel("UMAP dimension 2")

plt.tight_layout()
plt.savefig("./V3/plots/molecular_distribution_additives.png", dpi=600)

