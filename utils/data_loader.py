import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils.tools import Graph_data_generator, get_statistical_values
from sklearn.model_selection import train_test_split
from rdkit import Chem
import numpy as np

class MoleculeDataset(Dataset):
    def __init__(self, labeled_path, unlabeled_path, searching_space_path, analysis=True, cross_validate=True, 
                 embedding_visual=False, is_baseline=False):
        self.labeled_data_df = pd.DataFrame(pd.read_csv(labeled_path))
        self.unlabeled_data_df = pd.DataFrame(pd.read_csv(unlabeled_path))
        self.searching_space_df = pd.DataFrame(pd.read_csv(searching_space_path))

        self.analysis = analysis
        self.cross_validate = cross_validate
        self.embedding_visual = embedding_visual
        self.is_baseline = is_baseline

        if self.is_baseline:
            self.data = self.baseline_data()
            self.analysis = False
        else:
            self.data = self.load_data()

        if self.analysis:
            self.analysis_dataset(self.data)

    def load_data(self):
        labeled_cid_list = self.labeled_data_df['cid'].values.tolist()
        cids = list(set(self.searching_space_df['cid'].values))
        cids = [int(i) for i in cids]

        mass_mean, mass_std, vdw_mean, vdw_std, vdw_max, covalent_mean, covalent_std = self.get_mean_std_values(cids)

        data_list = []
        for cid in tqdm(cids, desc='Converting smiles data to graph data'):
            # get the graph data for each compound
            _, _, smile, _, _, _, _, label = self.read_from_one_call(cid)
            x, edge_index, edge_attr, label, n_nodes, n_edges, n_node_features, n_edge_features, descriptors = Graph_data_generator(smile, label, mass_mean, mass_std, vdw_mean, vdw_std, vdw_max, covalent_mean, covalent_std) # edge_attr: (n_edges, n_edge_features)
            if x == None:
                continue # if RDKit package can not convert smile into mol, we will drop this compound

            # get the mask for semi-supervised learning
            if cid in labeled_cid_list:
                graph_data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = label, mask=True, cid=cid, n_nodes = n_nodes, n_edges = n_edges, n_node_features = n_node_features, n_edge_features = n_edge_features, descriptors = descriptors.unsqueeze(0))
            else:
                graph_data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = label, mask=False, cid=cid, n_nodes = n_nodes, n_edges = n_edges, n_node_features = n_node_features, n_edge_features = n_edge_features, descriptors = descriptors.unsqueeze(0))
            
            data_list.append(graph_data)

        return data_list        
    
    def read_from_one_call(self, idx):
        formula = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'formula'].values[0])
        smile = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'SMILES'].values[0])
        fingerprint = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'fingerprint'].values[0])
        topological = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'topological'].values[0])
        weight = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'weight'].values[0])
        heavy_atom = str(self.searching_space_df.loc[self.searching_space_df['cid'] == float(idx), 'heavy_atom'].values[0])
        labeled_cid_list = self.labeled_data_df['cid'].values.tolist()
        if idx in labeled_cid_list:
            label = self.labeled_data_df.loc[self.labeled_data_df['cid'] == idx, 'label'].values[0]
        else:
            label = 2# stands for unlabeled data
        return idx, formula, smile, fingerprint, topological, weight, heavy_atom, label
    
    def data_split(self, data_list):
        train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

        return train_data, val_data, test_data
    
    def analysis_dataset(self, data_list):
        nodes, edges, nodes_feature, edges_feature = 0, 0, 0, 0
        for value in data_list:
            nodes += value.n_nodes
            edges += value.n_edges
            nodes_feature += value.n_node_features
            edges_feature += value.n_edge_features
        
        print('---------Here is the basic info of loaded dataset---------')
        print('number of nodes:', nodes)
        print('number of edges:', edges)
        print('nodes feature:', nodes_feature)
        print('edges feature:', edges_feature)
        print('number of degrees:', 2 * edges)
        print('avg degree:', 2 * edges / nodes)
        print('label rate:', len(self.save_labeled_data()) / self.__len__())
    
    def get_mean_std_values(self, cids):
        total_all_masses = []
        total_all_vdw = []
        total_all_covalent = []
        for cid in tqdm(cids, desc='Get some statistical values of data'):
            _, _, smile, _, _, _, _,_ = self.read_from_one_call(cid)
            all_masses, all_vdw, all_covalent = get_statistical_values(smile)
            if all_masses == None:
                continue

            total_all_masses += all_masses
            total_all_vdw += all_vdw
            total_all_covalent += all_covalent
        
        mass_mean, mass_std = np.mean(total_all_masses), np.std(total_all_masses)
        vdw_mean, vdw_std, vdw_max = np.mean(total_all_vdw), np.std(total_all_vdw), max(total_all_vdw)
        covalent_mean, covalent_std = np.mean(total_all_covalent), np.std(total_all_covalent)

        return mass_mean, mass_std, vdw_mean, vdw_std, vdw_max, covalent_mean, covalent_std
    
    def save_labeled_data(self):
        labeled_data_list = []
        for data in self.data:
            if data.y != 2:
                labeled_data_list.append(data)
        
        return labeled_data_list
    
    # smiles data
    def baseline_data(self):
        cids = list(set(self.searching_space_df['cid'].values))
        cids = [int(i) for i in cids]
        baseline_data = []
        for cid in tqdm(cids):
            _, _, smile, _, _, _, _, label = self.read_from_one_call(cid)
            smile = str(smile)
            label = int(label)
            if label != 2:
                data = Data(smile = smile, y = label)
                baseline_data.append(data)
        
        return baseline_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

