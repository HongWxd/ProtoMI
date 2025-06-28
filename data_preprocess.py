import pickle
from utils.data_loader import MoleculeDataset
import itertools
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# paths
labeled_path = './data/labeled_data.csv'
unlabeled_path = './data/unlabeled_data.csv'
searching_space_path = './data/searching_space_data.csv'
train_data_path = './data/train_data.pkl'
val_data_path = './data/val_data.pkl'
test_data_path = './data/test_data.pkl'
all_data_path = './data/all_data.pkl'

# load the dataset
dataset = MoleculeDataset(labeled_path, unlabeled_path, searching_space_path, is_baseline=True, load_flag=True, load_descriptors=True)
norm_MD, norm_VO, norm_YSS, norm_normal = dataset.norm_MD, dataset.norm_VO, dataset.norm_YSS, dataset.norm_normal

with open('./data/norm_MD.pkl', 'wb') as file:
    pickle.dump(dataset.norm_MD, file)
with open('./data/norm_VO.pkl', 'wb') as file:
    pickle.dump(dataset.norm_VO, file)
with open('./data/norm_YSS.pkl', 'wb') as file:
    pickle.dump(dataset.norm_YSS, file)
with open('./data/norm_normal.pkl', 'wb') as file:
    pickle.dump(dataset.norm_normal, file)

with open('./data/all_data.pkl', 'wb') as file:
    pickle.dump(dataset.data, file)

with open('./data/baseline_data.pkl', 'wb') as file:
    pickle.dump(dataset.baseline_data, file)



# labeled_data = dataset.save_labeled_data()
# with open('./data/baseline_data.pkl', 'wb') as f:
#     pickle.dump(dataset.data, f)
# with open('./data/all_data_descriptors_v2.pkl', 'wb') as file:
#     pickle.dump(dataset.data, file)

# with open('./data/labeled_data.pkl', 'wb') as file:
#     pickle.dump(labeled_data, file)








# preprocess the graph pairs for matric learning
# with open('./data/all_data.pkl', 'wb') as file:
#     all_data = pickle.load(dataset.data)
    
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


