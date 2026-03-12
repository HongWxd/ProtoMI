import json
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs
from rdkit import Chem
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score, 
    adjusted_mutual_info_score, 
    fowlkes_mallows_score,
    v_measure_score,
    homogeneity_score,
    completeness_score
)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt


def show_gnn_fp_consistency_results(additives_names, embeddings, best_k):
    with open('./V3/processed_data/additives.json', "r", encoding="utf-8") as f:
            additives_data = json.load(f)

    smiles_list = [additives_data[i]['smiles'] for i in additives_names]
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fps = []
    valid_smiles = []
    for smi in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = morgan_gen.GetFingerprint(mol)  
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_smiles.append(smi)
    fps = np.array(fps)

    Z_fp = linkage(fps, method='ward')
    Z_gnn = linkage(embeddings, method='ward')

    labels_fp = fcluster(Z_fp, t=best_k, criterion='maxclust')
    labels_gnn = fcluster(Z_gnn, t=best_k, criterion='maxclust')

    ari = adjusted_rand_score(labels_fp, labels_gnn)
    nmi = normalized_mutual_info_score(labels_fp, labels_gnn)
    ami = adjusted_mutual_info_score(labels_fp, labels_gnn)
    fmi = fowlkes_mallows_score(labels_fp, labels_gnn)
    v_measure = v_measure_score(labels_fp, labels_gnn)
    homogeneity = homogeneity_score(labels_fp, labels_gnn)
    completeness = completeness_score(labels_fp, labels_gnn)

    dist_fp = squareform(pdist(fps, metric='euclidean'))
    dist_gnn = squareform(pdist(embeddings, metric='euclidean'))
    spearman_corr, _ = spearmanr(dist_fp.ravel(), dist_gnn.ravel())

    print("==== Cluster Consistency Analysis Results ====")
    print(f"ARI:          {ari:.4f}")
    print(f"NMI:          {nmi:.4f}")
    print(f"AMI:          {ami:.4f}")
    print(f"FMI:          {fmi:.4f}")
    print(f"V-Measure:    {v_measure:.4f}")
    print(f"Homogeneity:  {homogeneity:.4f}")
    print(f"Completeness: {completeness:.4f}")
    print(f"Spearman Corr (Distance Structure): {spearman_corr:.4f}")


def plot_hierarchical_cluster_dendrogram(Z, additives_names):
    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=additives_names)
    plt.title("Hierarchical Clustering of Samples")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("./V3/plots/positive_samples_hierarchical_clustering.png", dpi=600)

def plot_cluster_distribution_UMAP(best_k, best_labels, embeddings, trial, args):
    plt.figure(figsize=(8,6))
    for i in range(1, best_k+1):
        plt.scatter(embeddings[best_labels==i, 0], embeddings[best_labels==i, 1], s=40, label=f"Cluster {i}", alpha=0.7)
    plt.legend()
    plt.title(f"UMAP Projection (Best Clusters = {best_k})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(f"./{args.save_path}/positive_samples_umap_best_cluster_trial_{trial}.png", dpi=600)