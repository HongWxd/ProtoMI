import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils.tools import Graph_data_generator
from sklearn.model_selection import train_test_split


class Dataset(Dataset):
    def __init__(self, labeled_path, unlabeled_path, searching_space_path, analysis=False):
        self.labeled_data_df = pd.DataFrame(pd.read_csv(labeled_path))
        self.unlabeled_data_df = pd.DataFrame(pd.read_csv(unlabeled_path))
        self.searching_space_df = pd.DataFrame(pd.read_csv(searching_space_path))

        self.analysis = analysis

    def load_data(self):
        labeled_cid_list = self.labeled_data_df['cid'].values.tolist()
        cids = self.searching_space_df['cid'].values.tolist()
        cids = [int(i) for i in cids]

        mask_dict = {}
        data_list = []
        for cid in tqdm(cids, desc='Converting smiles data to graph data'):
            # get the graph data for each compound
            _, _, smile, _, _, _, _, label = self.__getitem__(cid)
            if smile == None:
                continue # if RDKit package can not convert smile into mol, we will drop this compound

            x, edge_index, edge_attr, label, n_nodes, n_edges, n_node_features, n_edge_features = Graph_data_generator(smile, label) # edge_attr: (n_edges, n_edge_features)
            graph_data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = label, cid=cid, n_nodes = n_nodes, n_edges = n_edges, n_node_features = n_node_features, n_edge_features = n_edge_features)
            data_list.append(graph_data)

            # get the mask for semi-supervised learning
            if cid in labeled_cid_list:
                mask_dict[cid] = True
            else:
                mask_dict[cid] = False
        
        # split the dataset into train_data, val_data, and test_data
        train_dataset, val_dataset, test_dataset = self.data_split(data_list)

        if self.analysis:
            self.analysis_dataset(data_list)

        return train_dataset, val_dataset, test_dataset, mask_dict
    
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

    def __len__(self):
        return len(self.unlabeled_data_df) + len(self.labeled_data_df)
    
    def __getitem__(self, idx):
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
            label = -1
        return idx, formula, smile, fingerprint, topological, weight, heavy_atom, label
