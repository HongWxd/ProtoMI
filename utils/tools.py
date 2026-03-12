import numpy as np
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.utils.class_weight import compute_class_weight
import matminer.featurizers.composition as mm_composition
import matminer.featurizers.structure as mm_structure
from pymatgen.core import Composition
from torch_geometric.utils import subgraph
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score


device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

def get_atom_features(atom, mass_mean, mass_std, vdw_mean, vdw_std, covalent_mean, covalent_std,
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    node_features_labels = []
    # define list of permitted atoms
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms) # 43
    node_features_labels += permitted_list_of_atoms

    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"]) # 6
    node_features_labels += ['n_heavy_neighbors_0', 'n_heavy_neighbors_1', 'n_heavy_neighbors_2', 'n_heavy_neighbors_3', 'n_heavy_neighbors_4', "n_heavy_neighbors_MoreThanFour"]

    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]) # 8
    node_features_labels += ['formal_charge_-3', 'formal_charge_-2', 'formal_charge_-1', 'formal_charge_0', 'formal_charge_1', 'formal_charge_2', 'formal_charge_3', "formal_charge_Extreme"]

    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]) # 7
    node_features_labels += ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]

    is_in_a_ring_enc = [int(atom.IsInRing())] # 1
    node_features_labels += ['is_in_a_ring'] * len(is_in_a_ring_enc)

    is_aromatic_enc = [int(atom.GetIsAromatic())] # 1
    node_features_labels += ['is_aromatic'] * len(is_aromatic_enc)

    atomic_mass_scaled = [float((atom.GetMass() - mass_mean)/mass_std)] # 1
    node_features_labels += ['atomic_mass_scaled'] * len(atomic_mass_scaled)

    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - vdw_mean)/vdw_std)]  # 1
    node_features_labels += ['vdw_radius_scaled'] * len(vdw_radius_scaled)

    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - covalent_mean)/covalent_std)] # 1
    node_features_labels += ['covalent_radius_scaled'] * len(covalent_radius_scaled)

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        node_features_labels += ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]
        atom_feature_vector += chirality_type_enc # 4
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"]) # 6
        node_features_labels += ['n_hydrogens_0', 'n_hydrogens_1', 'n_hydrogens_2', 'n_hydrogens_3', 'n_hydrogens_4', "n_hydrogens_MoreThanFour"]
        atom_feature_vector += n_hydrogens_enc
    
    return np.array(atom_feature_vector) 

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    edge_features_labels = []
    # compute bond features
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    edge_features_labels += ['bond_type_single', 'bond_type_double', 'bond_type_triple', 'bond_type_aromatic']
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    edge_features_labels += ['bond_is_conjugated'] * len(bond_is_conj_enc)

    bond_is_in_ring_enc = [int(bond.IsInRing())]
    edge_features_labels += ['bond_is_in_ring'] * len(bond_is_in_ring_enc)

    bond_dir_enc = one_hot_encoding(str(bond.GetBondDir()), ["NONE", "BEGINWEDGE", "BEGINDASH", "ENDDOWNRIGHT", "ENDUPRIGHT"])
    edge_features_labels += ["bond_dir_none", "bond_dir_beginwedge", "bond_dir_begindash", "bond_dir_enddownright", "bond_dir_endupright"]

    bond_is_aromatic_enc = [int(bond.GetIsAromatic())]
    edge_features_labels += ['bond_is_aromatic'] * len(bond_is_aromatic_enc)

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc + bond_dir_enc + bond_is_aromatic_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        edge_features_labels += ["stereo_type_stereoz", "stereo_type_stereoe", "stereo_type_stereoany", "stereo_type_stereonone"]
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)

def get_reproted_descriptor(formula, mol):
    try:
        comp = Composition(formula)
    except:
        return None, None, None, None
    md_featurizer = mm_composition.Meredig()
    MD_descriptor = md_featurizer.featurize(comp)

    # os_featurizer = mm_composition.OxidationStates()
    # OS_descriptor = os_featurizer.featurize(comp)

    # sc_featurizer = mm_structure.StructuralComplexity()

    vo_featurizer = mm_composition.ValenceOrbital()
    VO_descriptor = vo_featurizer.featurize(comp)

    yss_featurizer = mm_composition.YangSolidSolution()
    YSS_descriptor = yss_featurizer.featurize(comp)

    Normal_descriptors = getMolDescriptors(mol)

    return Normal_descriptors, MD_descriptor, VO_descriptor, YSS_descriptor

def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
        missingVal is used if the descriptor cannot be calculated
    '''
    res = []
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res.append(val)
    return res

def Graph_data_generator(x_smiles, mass_mean, mass_std, vdw_mean, vdw_std, vdw_max, covalent_mean, covalent_std):
    # convert SMILES to RDKit mol object   
    mol = Chem.MolFromSmiles(x_smiles)
    if mol == None:
        return None, None, None, None, None, None, None

    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2*mol.GetNumBonds()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0), mass_mean, mass_std, vdw_mean, vdw_std, covalent_mean, covalent_std))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

    # construct node feature matrix X of shape (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))

    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom, mass_mean, mass_std, vdw_mean, vdw_std, covalent_mean, covalent_std)
        
    X = torch.tensor(X, dtype = torch.float)
    
    # construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim = 0) # (2, n_edges)
    
    # construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))
    
    for (k, (i,j)) in enumerate(zip(rows, cols)):
        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
    
    EF = torch.tensor(EF, dtype = torch.float)
    
    # construct label tensor
    # y_tensor = torch.tensor(np.array([y]), dtype = torch.long)
    
    # construct Pytorch Geometric data object and append to data list
    x = X
    edge_index = E
    edge_attr = EF
    # label = y_tensor
    n_nodes = n_nodes
    n_edges = n_edges
    n_node_features = n_node_features
    n_edge_features = n_edge_features

    return x, edge_index, edge_attr, n_nodes, n_edges, n_node_features, n_edge_features

def get_statistical_values(x_smiles):
    mol = Chem.MolFromSmiles(x_smiles)
    if mol == None:
        return None, None, None, None

    all_masses = []
    all_vdw = []
    all_covalent = []
    for atom in mol.GetAtoms():
        all_masses.append(float(atom.GetMass()))
        all_vdw.append(float(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())))
        all_covalent.append(float(Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())))

    return mol, all_masses, all_vdw, all_covalent


def training_data_analysis(fold, train_data, test_data):
    train_label_set = [i for i in train_data if i.mask == True]
    train_1 = len([i for i in train_label_set if i.y == 1])
    train_0 = len([i for i in train_label_set if i.y == 0])

    test_label_set = [i for i in test_data if i.mask == True]
    test_1 = len([i for i in test_label_set if i.y == 1])
    test_0 = len([i for i in test_label_set if i.y == 0])

    print(f'Fold {fold} | train 1: {(train_1) / (train_1 + train_0):.4f} | train 0: {train_0 / (train_1 + train_0):.4f} | test 1: {test_1 / (test_1 + test_0):.4f} | test 0: {test_0 / (test_1 + test_0):.4f} ')

def greedy_select_facilities(embeddings, k):
    # Greedy Max-Min selection of k medoids
    selected = []
    remaining = list(range(len(embeddings)))
    selected.append(torch.randint(0, len(embeddings), (1,)).item())

    for _ in range(1, k):
        dists = torch.stack([
            torch.norm(embeddings[i] - embeddings[selected], dim=1).min()
            for i in remaining
        ])
        next_idx = remaining[dists.argmax().item()]
        selected.append(next_idx)
        remaining.remove(next_idx)

    return embeddings[selected]

def facility_location_loss(embeddings, labels, gamma=1.0):
    # embeddings: [B, D], labels: [B]
    device = embeddings.device
    unique_labels = labels.unique()

    # Ground truth score
    gt_loss = 0.0
    for lbl in unique_labels:
        cluster = embeddings[labels == lbl]
        if cluster.shape[0] <= 1:
            continue
        dists = torch.cdist(cluster, cluster)
        medoid_idx = dists.sum(dim=1).argmin()
        medoid = cluster[medoid_idx]
        gt_loss += torch.norm(cluster - medoid, dim=1).sum()
    gt_loss = -gt_loss

    # Approximate worst clustering (greedy)
    S = greedy_select_facilities(embeddings, k=len(unique_labels))
    dist_mat = torch.cdist(embeddings, S)
    min_dist = dist_mat.min(dim=1)[0]
    F_S = -min_dist.sum()

    # NMI term
    pred_labels = dist_mat.argmin(dim=1).detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()
    delta = 1.0 - normalized_mutual_info_score(true_labels, pred_labels)

    loss = torch.clamp(F_S + gamma * delta - gt_loss, min=0.0)
    return loss

def imbalanced_weights(train_data, train_loader, epoch, device):
    total_masked = 0
    label_0 = 0
    label_1 = 0
    for data in train_loader:
        total_masked += int(data.mask.sum())
        label_0 += (data.y == 0).sum().item()
        label_1 += (data.y == 1).sum().item()
    print(f"[Epoch {epoch}] Train set labeled (mask=True): {total_masked} | label 0: {label_0 / total_masked:.4f} | label 1: {label_1 / total_masked:.4f}")

    y = []
    for data in train_data:
        y.append(data.y.item())
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    return total_masked, weights

def plot_PCL_Trials_SC(total_sc_scores, trials):
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, trials+1), total_sc_scores, marker='o', label='Silhoutte Score', color='tab:orange')
    plt.xlabel('PCL Trials')
    plt.ylabel('Silhoutte Core')
    plt.title(f'Prototye Contrastive Learning Trials vs Silhoutte Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./figs/PCL_Trials_SC.png', dpi=600)


def plot_train_loss(num_epochs, train_loss, model, training_types):
    epochs = list(range(1, num_epochs+1))

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_loss, marker='o', label='Train Loss', color='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model} Train Loss Curve')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./figs/{model}_{training_types}_train_loss_curve_{num_epochs}.png', dpi=600)


def perturb_edges(data, device, perturb_ratio=0.1):
    edge_index = data.edge_index.clone()
    num_edges = edge_index.size(1)
    num_nodes = data.x.size(0)

    num_delete = int(num_edges * perturb_ratio / 2)
    mask = torch.ones(num_edges, dtype=torch.bool)
    del_indices = random.sample(range(num_edges), num_delete)
    mask[del_indices] = False
    edge_index = edge_index[:, mask]

    num_add = num_delete
    new_edges = torch.randint(0, num_nodes, (2, num_add), device=device)
    edge_index = torch.cat([edge_index, new_edges], dim=1)

    data.edge_index = edge_index
    return data

def info_nce_loss(z1, z2, temperature=0.5):
    """
    z1, z2: shape [N, d]
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    N = z1.size(0)
    
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(N).to(z1.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def tanimoto_matrix(X):
    intersection = X @ X.T
    bit_sum = X.sum(axis=1)
    union = bit_sum[:, None] + bit_sum[None, :] - intersection
    union = np.where(union == 0, 1, union)
    sim = intersection / union
    return sim

def try_multiple_cluster_combinations(Z, all_embeddings, pos_additives_names, args):
    possible_clusters = range(3, args.max_cluster+1)
    best_score = -1
    best_k = None
    best_labels = None
    best_embeddings = None
    best_names = None
    best_Z = None

    for k in possible_clusters:
        cluster_labels = fcluster(Z, t=k, criterion='maxclust')

        # filter the outliers in each cluster
        filtered_embeddings, filtered_labels, filtered_indices = filter_cluster_outliers(all_embeddings, cluster_labels)

        try:
            Z_filt = linkage(filtered_embeddings, method='average', metric='cosine')
            score = silhouette_score(filtered_embeddings, filtered_labels, metric='cosine')
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = filtered_labels
                best_embeddings = filtered_embeddings
                if args.retrain_usl:
                    best_names = None
                else:
                    best_names = np.array(pos_additives_names)[filtered_indices].tolist()
                best_Z = Z_filt
        except Exception as e:
            print(f"k={k} failed: {e}")

    print(f"\n✅ best cluster number: {best_k}, average silhouette score: {best_score:.4f}")

    return best_k, best_labels, best_embeddings, best_names, best_Z

def filter_cluster_outliers(embeddings, cluster_labels, alpha=1.5):
    filtered_embeddings = []
    filtered_labels = []
    filtered_indices = []

    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        idx = np.where(cluster_labels == cluster_id)[0]
        cluster_emb = embeddings[idx]

        if len(cluster_emb) < 2:
            continue

        centroid = np.mean(cluster_emb, axis=0)
        distances = np.linalg.norm(cluster_emb - centroid, axis=1)

        # threshold = mean + α * std
        d_mean = distances.mean()
        d_std = distances.std()
        threshold = d_mean + alpha * d_std

        mask = distances <= threshold

        filtered_embeddings.append(cluster_emb[mask])
        filtered_labels.append(cluster_labels[idx][mask])
        filtered_indices.append(idx[mask])

    filtered_embeddings = np.vstack(filtered_embeddings)
    filtered_labels = np.concatenate(filtered_labels)
    filtered_indices = np.concatenate(filtered_indices)
    return filtered_embeddings, filtered_labels, filtered_indices