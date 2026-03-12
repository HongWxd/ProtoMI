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

with open('./data/all_data_test.pkl', 'wb') as file:
    pickle.dump(dataset.data, file)
