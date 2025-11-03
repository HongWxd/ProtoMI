import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from model import GCN, GINE, ProjectionHead
from tqdm import tqdm
import time
import pickle
from utils.tools import plot_train_loss
from utils.data_loader_CL import ContrastiveGraphDataset, contrastive_collate_fn
from utils.graph_augmentation import Graph_Augmentation_Helper
from sklearn.model_selection import KFold
import numpy as np
import torch.nn.functional as F
import warnings
from torch_geometric.data import Batch

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--ratio', type=int, default=15, help='negative samples ratio')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
parser.add_argument('--models', type=str, default='GINE', help='Training models')
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold of self training')
parser.add_argument('--warm_up_epoch', type=int, default=30, help='Self training warm up epoch period')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension of attention')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
parser.add_argument('--desp_dim', type=int, default=217, help='Number of descriptors')

# graph augmentation configs
parser.add_argument('--aug_types', type=str, default='all', help='augmentation types')
parser.add_argument('--shuffle_ratio', type=float, default=0.2, help='shuffle ratio')
parser.add_argument('--node_drop_ratio', type=float, default=0.2, help='node drop ratio')
parser.add_argument('--noise_ratio', type=float, default=0.2, help='noise_ratio')
parser.add_argument('--noise_std', type=float, default=0.1, help='noise_std')
parser.add_argument('--edge_drop_ratio', type=float, default=0.1, help='edge drop ratio')
parser.add_argument('--edge_add_ratio', type=float, default=0.05, help='edge add ratio')
parser.add_argument('--alpha', type=float, default=0.15, help='PPR alpha')
parser.add_argument('--PPR_drop_ratio', type=float, default=0.2, help='PPR_drop_ratio')
parser.add_argument('--PPR_add_ratio', type=float, default=0.2, help='PPR_add_ratio')
parser.add_argument('--K', type=int, default=10, help='PPR K')
parser.add_argument('--random_state', type=int, default=42, help='data split random seed')
parser.add_argument('--test_size', type=float, default=0.2, help='test set size')



args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


# data preparation
with open('./data/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

positive_samples = all_data[:126]
unlabeled_samples = all_data[126:]
graph_aug_helper = Graph_Augmentation_Helper(positive_samples, args)
pos_train_samples, pos_test_samples = graph_aug_helper.train_test_split_positive_samples()
print(pos_train_samples[0])

















# plot_train_loss(args.epoch, train_loss, args.models)

# # Save the model
# torch.save(encoder.state_dict(), f'./checkpoints/CL_encoder_{args.models}.pth')
