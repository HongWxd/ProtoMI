import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils.tools import Graph_data_generator

class Dataset(Dataset):
    def __init__(self, labeled_path, unlabeled_path, searching_space_path):
        self.labeled_data_df = pd.DataFrame(pd.read_csv(labeled_path))
        self.unlabeled_data_df = pd.DataFrame(pd.read_csv(unlabeled_path))
        self.searching_space_df = pd.DataFrame(pd.read_csv(searching_space_path))

    def load_data(self):
        labeled_cid_list = self.labeled_data_df['cid'].values.tolist()
        cids = self.searching_space_df['cid'].values.tolist()
        cids = [int(i) for i in cids]

        graph_dict = {}
        mask_dict = {}
        for cid in tqdm(cids, desc='Converting smiles data to graph data'):
            # get the graph data for each compound
            _, _, smile, _, _, _, _, label = self.__getitem__(cid)
            try:
                x, edge_index, edge_attr, y = Graph_data_generator(smile, label) # edge_attr: (n_edges, n_edge_features)
            except:
                continue # while RDKit package can not convert smile into mol will drop this compound
            graph_data = Data(x = x, edge_index = edge_index, y = y)
            graph_dict[cid] = graph_data

            # get the mask
            if cid in labeled_cid_list:
                mask_dict[cid] = True
            else:
                mask_dict[cid] = False
            
        return graph_dict, mask_dict

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
            label = self.labeled_data_df.loc[self.labeled_data_df['cid'] == idx, 'labels'].values[0]
        else:
            label = -1
        return idx, formula, smile, fingerprint, topological, weight, heavy_atom, label


labeled_path = './data/labeled_data.csv'
unlabeled_path = './data/unlabeled_data.csv'
searching_space_path = './data/searching_space_data.csv'
Dataset = Dataset(labeled_path, unlabeled_path, searching_space_path)
graph_dict, mask_dict = Dataset.load_data()
for graph, mask in zip(graph_dict, mask_dict):
    print(graph, mask.values())

