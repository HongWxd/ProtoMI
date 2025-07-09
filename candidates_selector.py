import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

labeled_df = pd.DataFrame(pd.read_csv('./data/labeled_UMAP.csv'))
candidates_df = pd.DataFrame(pd.read_csv('./data/kmeans_clusters.csv'))

cluster_10 = labeled_df.loc[labeled_df['cluster'] == 10, 'smiles'].values.tolist()
candidates_10 = candidates_df.loc[candidates_df['cluster'] == 10, 'smiles'].values.tolist()


def sorted_by_Tanimoto(cluster_number):
    # labeled molecules
    cluster_smiles = labeled_df.loc[labeled_df['cluster'] == cluster_number, 'smiles'].values.tolist()
    cluster_mols = [Chem.MolFromSmiles(s) for s in cluster_smiles]
    cluster_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in cluster_mols]

    # candidates molecules
    candidate_smiles = candidates_df.loc[candidates_df['cluster'] == cluster_number, 'smiles'].values.tolist()
    candidate_mols = [Chem.MolFromSmiles(s) for s in candidate_smiles]
    candidate_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in candidate_mols]

    # Calculate the similarity matrix of all members and candidate molecules within the cluster.
    similarity_matrix = np.zeros((len(cluster_fps), len(candidate_fps)))
    # print(similarity_matrix.shape)

    for i, cluster_fp in enumerate(cluster_fps):
        for j, candidate_fp in enumerate(candidate_fps):
            sim = DataStructs.FingerprintSimilarity(cluster_fp, candidate_fp)
            similarity_matrix[i, j] = sim

    # For candidate molecules, calculate the "average similarity" of the clusters.
    avg_similarities = similarity_matrix.mean(axis=0)
    max_similarities = similarity_matrix.max(axis=0)
    avg_distances = 1 - avg_similarities
    max_distances = 1 - max_similarities

    # sort by distances
    sorted_idx_avg = np.argsort(avg_distances)
    sorted_idx_max = np.argsort(max_distances)

    avg_results_df = pd.DataFrame()
    cids = []
    smiles = []
    ranks = []
    distances = []
    for rank, idx in enumerate(sorted_idx_avg, 1):
        cid = candidates_df.loc[candidates_df['smiles'] == candidate_smiles[idx], 'cid'].values[0]
        cids.append(cid)
        smiles.append(candidate_smiles[idx])
        ranks.append(rank)
        distances.append(avg_distances[idx])
        # print(f"{rank}. {candidate_smiles[idx]} (distance = {avg_distances[idx]:.4f})")
    avg_results_df['cid'] = cids
    avg_results_df['smiles'] = smiles
    avg_results_df['rank'] = ranks
    avg_results_df['distance'] = distances

    max_results_df = pd.DataFrame()
    cids = []
    smiles = []
    ranks = []
    distances = []
    for rank, idx in enumerate(sorted_idx_max, 1):
        cid = candidates_df.loc[candidates_df['smiles'] == candidate_smiles[idx], 'cid'].values[0]
        cids.append(cid)
        smiles.append(candidate_smiles[idx])
        ranks.append(rank)
        distances.append(max_distances[idx])
        # print(f"{rank}. {candidate_smiles[idx]} (distance = {max_distances[idx]:.4f})")
    max_results_df['cid'] = cids
    max_results_df['smiles'] = smiles
    max_results_df['rank'] = ranks
    max_results_df['distance'] = distances

    # save results
    avg_results_df.to_csv(f'./data/cluster_{cluster_number}_avg_distances.csv', index=False)
    max_results_df.to_csv(f'./data/cluster_{cluster_number}_max_distances.csv', index=False)

sorted_by_Tanimoto(cluster_number=3)
sorted_by_Tanimoto(cluster_number=6)
sorted_by_Tanimoto(cluster_number=10)