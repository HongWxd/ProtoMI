import pickle
from utils.data_loader import MoleculeDataset
import itertools
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# paths
labeled_path = './V3/processed_data/additives.json'
searching_space_path = './data/searching_space_data_V2.csv'
all_data_path = './data/all_data.pkl'

# load the dataset
dataset = MoleculeDataset(labeled_path, searching_space_path)

with open('./data/all_data_test.pkl', 'wb') as file:
    pickle.dump(dataset.data, file)
