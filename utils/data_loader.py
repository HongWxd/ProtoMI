import torch
import pandas as pd
import numpy as np
import json
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils.tools import Graph_data_generator, get_statistical_values, perturb_edges
from sklearn.model_selection import train_test_split
from utils.graph_augmentation import Graph_Augmentation_Helper
from sklearn.decomposition import PCA

class MoleculeDataset(Dataset):
    def __init__(self, additives_data, searching_space_df):
        self.reported_smiles = [i['smiles'] for i in additives_data.values()]
        self.reported_formulas = additives_data.keys()
        self.searching_space_df = searching_space_df

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
        print('Generate additive-ID mapping file.')

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

        # # filter the redundent features
        # data_list = self.features_reduction(data_list, 'node')
        # data_list = self.features_reduction(data_list, 'edge')
        
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
        low_contrib_mask = np.all(np.abs(loadings_subset) < 0.01, axis=0)
        filtered_indices = np.where(~low_contrib_mask)[0]

        for data in data_list:
            if hasattr(data, 'x') and data.x is not None:
                data.x = data.x[:, filtered_indices]

        return data_list

    def analysis_dataset(self, data_list):

        atoms_list = []
        bonds_list = []
        avg_degree_list = []
        density_list = []
        entropy_list = []

        for data in data_list:

            n_nodes = data.n_nodes
            n_edges = data.n_edges

            # ---------------------
            # Atoms
            # ---------------------
            atoms_list.append(n_nodes)

            # ---------------------
            # Bonds
            # ---------------------
            bonds_list.append(n_edges)

            # ---------------------
            # Average degree
            # <k>=2E/N
            # ---------------------
            avg_degree = 2 * n_edges / n_nodes
            avg_degree_list.append(avg_degree)

            # ---------------------
            # Graph density
            # D=2E/[N(N-1)]
            # undirected graph
            # ---------------------
            density = 2 * n_edges / (n_nodes * (n_nodes - 1) + 1e-12)
            density_list.append(density)

            # ---------------------
            # Degree entropy
            # H=-Σp(k)ln(p(k))
            # ---------------------
            row = data.edge_index[0].cpu().numpy()
            degree = np.bincount(row, minlength=n_nodes)

            p = degree / (degree.sum() + 1e-12)

            entropy = -(p * np.log(p + 1e-12)).sum()

            entropy_list.append(entropy)


        # =========================
        # Convert to numpy
        # =========================
        atoms_list = np.array(atoms_list)
        bonds_list = np.array(bonds_list)
        avg_degree_list = np.array(avg_degree_list)
        density_list = np.array(density_list)
        entropy_list = np.array(entropy_list)


        print('--------- Dataset Statistics ---------')

        print(f'Atoms          : {atoms_list.mean():.2f} ± {atoms_list.std():.2f}')

        print(f'Bonds          : {bonds_list.mean():.2f} ± {bonds_list.std():.2f}')

        print(f'Average degree : {avg_degree_list.mean():.3f} ± {avg_degree_list.std():.3f}')

        print(f'Graph density  : {density_list.mean():.3f} ± {density_list.std():.3f}')

        print(f'Degree entropy : {entropy_list.mean():.3f} ± {entropy_list.std():.3f}')
        
    
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


def load_data(args):
    # load the year mapping file
    with open(args.year_mapping_path, 'r', encoding='utf-8') as f:
        year_mapping = json.load(f)
    
    # split training data by year
    if args.split_year == 'all':
        positive_num = year_mapping['n_positive']

    else:
        split_name = f'cutoff_{args.split_year}'

        positive_num = (
            year_mapping['cutoffs'][split_name]
            ['train_positive_end_idx']
            + 1
        )


    if args.method == 'morgan':
        # data preparation for graph data
        with open(args.data_path, 'rb') as f:
            all_data = pickle.load(f)

        unlabeled_samples_graph = all_data[positive_num:]

        # load smiles data for Morgan fingerprint baseline
        with open(args.additive_json_path, 'r') as f:
            positive_data = json.load(f)

        searching_space = pd.read_csv(args.searching_space_path)

        positive_samples = []
        for i, (_, v) in enumerate(positive_data.items()):
            if i >= positive_num:
                break
            positive_samples.append(v['smiles'])
        
        unlabeled_samples = [{'id': int(id), 'smiles': smile} for id, smile in zip(searching_space['cid'].values.tolist(), searching_space['SMILES'].values.tolist())]

        return positive_samples, unlabeled_samples, unlabeled_samples_graph

    else:
        # data preparation for graph data
        with open(args.data_path, 'rb') as f:
            all_data = pickle.load(f)

        positive_samples = all_data[:positive_num] # number of positive samples
        unlabeled_samples = all_data[positive_num:]

        graph_aug_helper = Graph_Augmentation_Helper(positive_samples, args)
        pos_train_samples, pos_test_samples = graph_aug_helper.train_test_split_positive_samples()
        unl_train_samples, unl_test_samples = train_test_split(unlabeled_samples, test_size=args.test_size, random_state=args.random_state)
        return positive_samples, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples
