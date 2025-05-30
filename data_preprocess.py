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
dataset = MoleculeDataset(labeled_path, unlabeled_path, searching_space_path)
graph_data_pairs = dataset.build_graph_pairs()
with open('./data/data_pairs.pkl', 'wb') as f:
    pickle.dump(graph_data_pairs, f)


# pairs = itertools.combinations(data_list, 2)
# all_pairs = []
# for pair in pairs:
#     if pair[0].y == pair[1].y == 1:
#         print(pair[0].y, pair[1].y)
#         all_pairs.append((pair[0], pair[1], 1))
#     else:
#         print(pair[0].y, pair[1].y)
#         all_pairs.append((pair[0], pair[1], 0))
    



