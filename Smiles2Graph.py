import pandas as pd
from utils.tools import Graph_data_generator

labeled_data_df = pd.DataFrame(pd.read_csv('./data/labeled_data.csv'))
unlabeled_data_df = pd.DataFrame(pd.read_csv('./data/unlabeled_data.csv'))
unlabeled_cid_list = unlabeled_data_df['cid'].values.tolist()
labeled_cid_list = labeled_data_df['cid'].values.tolist()

mask_dict = {} # get the mask dictionary for labeled and unlabeled data
for cid in labeled_cid_list:
    mask_dict[cid] = True
for cid in unlabeled_cid_list:
    mask_dict[int(cid)] = False

