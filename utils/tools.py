import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score
from ase import Atoms
from dscribe.descriptors import SOAP
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

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

    # define list of permitted atoms
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass() - mass_mean)/mass_std)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - vdw_mean)/vdw_std)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - covalent_mean)/covalent_std)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)

def get_SOAP_descriptor(mol, vdw_max):
    SOAP_mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(SOAP_mol)
    AllChem.UFFOptimizeMolecule(SOAP_mol)
    conf = SOAP_mol.GetConformer()
    positions = []
    symbols = []
    for atom in SOAP_mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        symbols.append(atom.GetSymbol())
    positions = np.array(positions)

    ase_mol = Atoms(symbols=symbols, positions=positions)

    species = list(set(symbols))
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=2*vdw_max,
        n_max=8,
        l_max=6,
    )
    soap_descriptor = soap.create(ase_mol)
    # print("SOAP shape:", soap_descriptor.shape)

    return soap_descriptor

def Graph_data_generator(x_smiles, y, mass_mean, mass_std, vdw_mean, vdw_std, vdw_max, covalent_mean, covalent_std):
    # convert SMILES to RDKit mol object   
    mol = Chem.MolFromSmiles(x_smiles)
    if mol == None:
        return None, None, None, None, None, None, None, None

    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2*mol.GetNumBonds()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0), mass_mean, mass_std, vdw_mean, vdw_std, covalent_mean, covalent_std))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

    # get descriptors
    soap_descriptor = get_SOAP_descriptor(mol)

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
    y_tensor = torch.tensor(np.array([y]), dtype = torch.long)
    
    # construct Pytorch Geometric data object and append to data list
    x = X
    edge_index = E
    edge_attr = EF
    label = y_tensor
    n_nodes = n_nodes
    n_edges = n_edges
    n_node_features = n_node_features
    n_edge_features = n_edge_features

    return x, edge_index, edge_attr, label, n_nodes, n_edges, n_node_features, n_edge_features

def get_statistical_values(x_smiles):
    mol = Chem.MolFromSmiles(x_smiles)
    if mol == None:
        return None, None, None

    all_masses = []
    all_vdw = []
    all_covalent = []
    for atom in mol.GetAtoms():
        all_masses.append(float(atom.GetMass()))
        all_vdw.append(float(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())))
        all_covalent.append(float(Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())))

    return all_masses, all_vdw, all_covalent

def unlabeled_weight(epoch, T1, T2):
        alpha = 0.0
        af = 3
        if epoch > T1:
            alpha = (epoch-T1) / (T2-T1)*af
            if epoch > T2:
                alpha = af
        return alpha

def self_training(model, data, loss, out, epoch, criterion, device, args):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
        probs = F.softmax(logits, dim=-1)
        confidence, pseudo_labels = probs.max(dim=1)

    model.train()
    high_conf_mask = (confidence > args.threshold) & (~data.mask)
    if high_conf_mask.sum() > 0:
        pseudo_labels = pseudo_labels.detach()
        pseudo_loss = criterion(out[high_conf_mask], pseudo_labels[high_conf_mask])
        # loss += pseudo_loss
        pseudo_samples = int(len(out[high_conf_mask]))

        data.y = data.y.clone()
        data.mask = data.mask.clone()
        data.y[high_conf_mask] = pseudo_labels[high_conf_mask]
        data.mask[high_conf_mask] = True
    
    return loss, data

# def update_training_loader(train_loader, model, device, args):
#     model.eval()
#     updated_dataset = []
#     for data in train_loader.dataset:
#         data = data.to(device)
#         if not hasattr(data, 'mask') or data.mask.sum().item() > 0:
#             updated_dataset.append(data.cpu())
#             continue

#         with torch.no_grad():
#             logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
#             probs = F.softmax(logits, dim=-1)
#             confidence, pseudo_labels = probs.max(dim=1)

#         high_conf_mask = (confidence > args.threshold)
#         if high_conf_mask.sum().item() > 0:
#             data.y = data.y.clone()
#             data.mask = data.mask.clone()
#             data.y[high_conf_mask] = pseudo_labels[high_conf_mask].detach().cpu()
#             data.mask[high_conf_mask] = True

#         updated_dataset.append(data.cpu())

#     print(f"[Update] Expanded labeled set with pseudo-labeled samples. New size: {sum(d.mask.sum().item() for d in updated_dataset)}")
#     return DataLoader(updated_dataset, batch_size=train_loader.batch_size, shuffle=True)

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

def plot_loss_acc(num_epochs, train_loss, total_test_loss, test_accuracy, fold):
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, marker='o', label='Train Loss')
    plt.plot(epochs, total_test_loss, marker='o', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracy, marker='o', color='gray', label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy Curve')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./figs/fold_{fold+1}_loss_acc_curve.png', dpi=600)


