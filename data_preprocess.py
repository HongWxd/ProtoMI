import pickle
from utils.data_loader import MoleculeDataset
import itertools
from tqdm import tqdm

# paths
labeled_path = './data/labeled_data.csv'
unlabeled_path = './data/unlabeled_data.csv'
searching_space_path = './data/searching_space_data.csv'
train_data_path = './data/train_data.pkl'
val_data_path = './data/val_data.pkl'
test_data_path = './data/test_data.pkl'
all_data_path = './data/all_data.pkl'

# load the dataset
dataset = MoleculeDataset(labeled_path, unlabeled_path, searching_space_path, is_baseline=False)
# with open('./data/baseline_data.pkl', 'wb') as f:
#     pickle.dump(dataset.data, f)



# preprocess the graph pairs for matric learning
with open('./data/all_data.pkl', 'rb') as file:
    all_data = pickle.load(dataset.data)
    
# graph_index_list = list(range(len(all_data)))
# index_pairs = itertools.combinations(graph_index_list, 2)
# n = len(all_data)
# pairs_len = n * (n-1) // 2
# all_pairs = []
# split = 0
# for pair, i in zip(index_pairs, tqdm(range(pairs_len))):
#     if all_data[pair[0]].y == all_data[pair[1]].y == 1:
#         all_pairs.append((pair[0], pair[1], 1))
#     else:
#         all_pairs.append((pair[0], pair[1], 0))
    
#     if len(all_pairs) >= 100000000:
#         split += 1
#         with open(f'/data/hwx/Boron/data_pairs_{split}.pkl', 'wb') as f:
#             pickle.dump(all_pairs, f)
        
#         all_pairs = []


