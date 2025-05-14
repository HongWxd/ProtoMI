import argparse
import pandas as pd
from data_loader import Dataset
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--labeled_path', type=str, default='./data/labeled_data.csv', help='Path for labeled data')
parser.add_argument('--unlabeled_path', type=str, default='./data/unlabeled_data.csv', help='Path for unlabeled data')
parser.add_argument('--searching_space_path', type=str, default='./data/searching_space_data.csv', help='Path for searching space data')
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

args = parser.parse_args()

# load the dataset
Dataset = Dataset(args.labeled_path, args.unlabeled_path, args.searching_space_path, analysis=True)
train_data, val_data, test_data, mask_dict = Dataset.load_data()
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

