import pickle
from utils.data_loader_CL import MoleculeDataset
import itertools
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# paths
labeled_path = './V3/processed_data/additives.json'
searching_space_path = './data/searching_space_data_V2.csv'
train_data_path = './data/train_data.pkl'
val_data_path = './data/val_data.pkl'
test_data_path = './data/test_data.pkl'
all_data_path = './data/all_data.pkl'

# load the dataset
dataset = MoleculeDataset(labeled_path, searching_space_path)
# norm_MD, norm_VO, norm_YSS, norm_normal = dataset.norm_MD, dataset.norm_VO, dataset.norm_YSS, dataset.norm_normal
# print(len(dataset.data), len(norm_normal), len(dataset.baseline_data))

# with open('./data/norm_MD.pkl', 'wb') as file:
#     pickle.dump(dataset.norm_MD, file)
# with open('./data/norm_VO.pkl', 'wb') as file:
#     pickle.dump(dataset.norm_VO, file)
# with open('./data/norm_YSS.pkl', 'wb') as file:
#     pickle.dump(dataset.norm_YSS, file)
# with open('./data/norm_normal.pkl', 'wb') as file:
#     pickle.dump(dataset.norm_normal, file)

with open('./data/all_data.pkl', 'wb') as file:
    pickle.dump(dataset.data, file)

# with open('./data/baseline_data.pkl', 'wb') as file:
#     pickle.dump(dataset.baseline_data, file)



# labeled_data = dataset.save_labeled_data()
# with open('./data/baseline_data.pkl', 'wb') as f:
#     pickle.dump(dataset.data, f)
# with open('./data/all_data_descriptors_v2.pkl', 'wb') as file:
#     pickle.dump(dataset.data, file)

# with open('./data/labeled_data.pkl', 'wb') as file:
#     pickle.dump(labeled_data, file)


