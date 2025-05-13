import pandas as pd
from data_loader import Dataset

labeled_path = './data/labeled_data.csv'
unlabeled_path = './data/unlabeled_data.csv'
searching_space_path = './data/searching_space_data.csv'
Dataset = Dataset(labeled_path, unlabeled_path, searching_space_path)
graph_dict, mask_dict = Dataset.load_data()
nodes, edges, nodes_feature, edges_feature = 0, 0, 0, 0
for key, value in graph_dict.items():
    nodes += value.n_nodes
    edges += value.n_edges
    nodes_feature += value.n_node_features
    edges_feature += value.n_edge_features
    print(key, value, value.n_node_features, value.n_edge_features)
print('nodes:', nodes)
print('edge:', edges)
print('nodes feature:', nodes_feature)
print('edges feature:', edges_feature)
print('degree:', 2 * edges)
print('avg degree:', 2 * edges / nodes)