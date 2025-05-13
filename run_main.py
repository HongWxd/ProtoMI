import pandas as pd
from data_loader import Dataset


labeled_path = './data/labeled_data.csv'
unlabeled_path = './data/unlabeled_data.csv'
searching_space_path = './data/searching_space_data.csv'
Dataset = Dataset(labeled_path, unlabeled_path, searching_space_path)
graph_dict, mask_dict = Dataset.load_data()

