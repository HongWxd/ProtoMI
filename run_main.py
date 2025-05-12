import pandas as pd
from utils.tools import Graph_data_generator
import pickle

labeled_data_df = pd.DataFrame(pd.read_csv('./data/labeled_data.csv'))
labeled_smiles = labeled_data_df['smiles'].values.tolist()
labeled_y = labeled_data_df['labels'].values.tolist()
labeled_data_list = Graph_data_generator(labeled_smiles, labeled_y)
with open('./data/labeled_graph_data.pkl', 'wb') as f:
    pickle.dump(labeled_data_list, f)

unlabeled_data_df = pd.DataFrame(pd.read_csv('./data/unlabeled_data.csv'))
unlabeled_smiles = unlabeled_data_df['smiles'].values.tolist()
unlabeled_y = [-1] * len(unlabeled_smiles)
unlabeled_data_list = Graph_data_generator(unlabeled_smiles, unlabeled_y)
with open('./data/unlabeled_graph_data.pkl', 'wb') as f:
    pickle.dump(unlabeled_data_list, f)

print((unlabeled_data_list))