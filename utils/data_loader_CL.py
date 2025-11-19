import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils.tools import Graph_data_generator, get_statistical_values, perturb_edges
import numpy as np
import json
import random
from sklearn.decomposition import PCA

class MoleculeDataset(Dataset):
    def __init__(self, labeled_path, searching_space_path):
        with open(labeled_path, 'r') as file:
            additives_data = json.load(file)
        
        self.reported_smiles = [i['smiles'] for i in additives_data.values()]
        self.reported_formulas = additives_data.keys()
        self.searching_space_df = pd.DataFrame(pd.read_csv(searching_space_path))

        self.cids = list(sorted(set(self.searching_space_df['cid'].values)))
        self.cids = [int(i) for i in self.cids]

        self.all_smiles = []
        self.id = []
        self.names = []
        # load all molecules smiles
        for smile, id, name in tqdm(zip(self.reported_smiles, list(range(len(self.reported_formulas))), self.reported_formulas), desc='Loading all reported molecules'):
            self.all_smiles.append(smile)
            self.id.append(int(id))
            self.names.append(name)

        additive_id_df = pd.DataFrame()
        additive_id_df['id'] = self.id
        additive_id_df['name'] = self.names
        additive_id_df.to_csv('./V3/processed_data/additive_id_mapping.csv', index=False)

        for cid in tqdm(self.cids, desc='Loading all molecules from searching space'):
            _, _, smile, _, _, _, _ = self.read_from_one_call(cid)
            self.all_smiles.append(smile)
            self.id.append(int(cid))

        self.data = self.load_data()
        
        self.analysis_dataset(self.data)

    def load_data(self):
        # for hard code normalization purpose
        mass_mean, mass_std, vdw_mean, vdw_std, vdw_max, covalent_mean, covalent_std = self.get_mean_std_values()

        data_list = []
        # add all molecules data
        for smile, id in zip(tqdm(self.all_smiles, desc='Converting all smiles data to graph data'), self.id):
            # get the graph data for each compound
            x, edge_index, edge_attr, n_nodes, n_edges, _, _ = Graph_data_generator(smile, mass_mean, mass_std, vdw_mean, vdw_std, vdw_max, covalent_mean, covalent_std) # edge_attr: (n_edges, n_edge_features)
            if x == None:
                continue # if RDKit package can not convert smile into mol, we will drop this molecule

            graph_data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, id = id, n_nodes = n_nodes, n_edges = n_edges)
            data_list.append(graph_data)
        
        # normalization for edge features
        all_edge_attrs = [data.edge_attr for data in data_list if data.edge_attr is not None]
        all_edge_attrs = torch.cat(all_edge_attrs, dim=0)

        edge_mean = all_edge_attrs.mean(dim=0, keepdim=True)
        edge_std = all_edge_attrs.std(dim=0, keepdim=True) + 1e-6

        for data in data_list:
            if data.edge_attr is not None:
                data.edge_attr = (data.edge_attr - edge_mean) / edge_std

        # filter the redundent features
        data_list = self.features_reduction(data_list, 'node')
        data_list = self.features_reduction(data_list, 'edge')
        
        return data_list        

    def read_from_one_call(self, idx):
        formula = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'formula'].values[0])
        smile = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'SMILES'].values[0])
        fingerprint = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'fingerprint'].values[0])
        topological = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'topological'].values[0])
        weight = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'weight'].values[0])
        heavy_atom = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'heavy_atom'].values[0])
        return idx, formula, smile, fingerprint, topological, weight, heavy_atom
    
    def features_reduction(self, data_list, features_type):
        all_features = []
        for data in data_list:
            if hasattr(data, 'x') and data.x is not None:
                if features_type == 'node':
                    all_features.append(data.x.numpy())
                elif features_type == 'edge':
                    all_features.append(data.edge_attr.numpy())

        X = np.vstack(all_features)

        pca = PCA(n_components=min(10, X.shape[1]))  
        pca.fit(X)

        loadings = pca.components_  # shape = [n_components, n_features]
        n_components = loadings.shape[0]

        top_k = min(10, n_components)
        loadings_subset = loadings[:top_k, :]
        low_contrib_mask = np.all(np.abs(loadings_subset) < 0.05, axis=0)
        filtered_indices = np.where(~low_contrib_mask)[0]

        for data in data_list:
            if hasattr(data, 'x') and data.x is not None:
                data.x = data.x[:, filtered_indices]

        return data_list

    def analysis_dataset(self, data_list):
        nodes, edges, nodes_feature, edges_feature = 0, 0, 0, 0
        for value in data_list:
            nodes += value.n_nodes
            edges += value.n_edges
            nodes_feature += value.x.shape[1]
            edges_feature += value.edge_attr.shape[1]
        
        print('---------Here is the basic info of loaded dataset---------')
        print('avg nodes:', nodes / self.__len__())
        print('number of edges:', edges / self.__len__())
        print('nodes feature:', nodes_feature / self.__len__())
        print('edges feature:', edges_feature / self.__len__())
        print('number of degrees:', 2 * edges / self.__len__())
        print('avg degree:', 2 * edges / nodes)
    
    def get_mean_std_values(self):
        total_all_masses = []
        total_all_vdw = []
        total_all_covalent = []
        for smile in tqdm(self.all_smiles, desc='Get some statistical values of data'):
            _, all_masses, all_vdw, all_covalent = get_statistical_values(smile)
            if all_masses == None:
                continue

            total_all_masses += all_masses
            total_all_vdw += all_vdw
            total_all_covalent += all_covalent
        
        mass_mean, mass_std = np.mean(total_all_masses), np.std(total_all_masses)
        vdw_mean, vdw_std, vdw_max = np.mean(total_all_vdw), np.std(total_all_vdw), max(total_all_vdw)
        covalent_mean, covalent_std = np.mean(total_all_covalent), np.std(total_all_covalent)

        return mass_mean, mass_std, vdw_mean, vdw_std, vdw_max, covalent_mean, covalent_std

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class ContrastiveGraphDataset(Dataset):
    def __init__(self, positive_samples, unlabeled_samples, ratio=5):
        self.positive_samples = positive_samples
        self.unlabeled_samples = unlabeled_samples
        self.ratio = ratio

    def __len__(self):
        return len(self.positive_samples)
    
    def __getitem__(self, idx):
        return self.positive_samples[idx]