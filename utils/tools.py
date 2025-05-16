import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

def get_atom_features(atom, 
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
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
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

def Graph_data_generator(x_smiles, y):
    # convert SMILES to RDKit mol object   
    mol = Chem.MolFromSmiles(x_smiles)
    if mol == None:
        return None, None, None, None, None, None, None, None

    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2*mol.GetNumBonds()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
    # print(smiles, n_nodes, n_edges, unrelated_smiles, unrelated_mol, n_node_features, n_edge_features)

    # construct node feature matrix X of shape (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))

    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)
        
    X = torch.tensor(X, dtype = torch.float)
    
    # construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim = 0)
    
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

def plot_loss_acc(num_epochs, train_loss, train_samples, total_test_loss, test_samples, test_accuracy, fold):
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss / train_samples, marker='o', label='Train Loss')
    plt.plot(epochs, total_test_loss / test_samples, marker='o', label='Test Loss')
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

def train(model, train_loader, device, optimizer, criterion):
    model.train()
    total_loss = 0
    total_samples = 0
    for data in train_loader:
        if data.mask.sum() == 0:
            continue

        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)

        loss = criterion(out[data.mask], data.y[data.mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += int(data.mask.sum())
    
    return total_loss, total_samples

def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_samples = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            if data.mask.sum() == 0:
                continue

            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out[data.mask], data.y[data.mask])

            pred = out.argmax(dim=1)
            all_preds.append(pred[data.mask].cpu())
            all_labels.append(data.y[data.mask].cpu())
            total_samples += int(data.mask.sum())
            total_loss += loss.item()
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')

    return accuracy, precision, recall, f1, total_samples, total_loss
