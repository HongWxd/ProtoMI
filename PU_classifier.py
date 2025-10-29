import torch
import argparse
import pandas as pd
from torch_geometric.loader import DataLoader
from model import PU_Classifier, PU_GCN
from tqdm import tqdm
import time
import pickle
from utils.tools import nnPU_loss, plot_train_loss
from sklearn.model_selection import KFold
import numpy as np
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=500, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
parser.add_argument('--models', type=str, default='PU_Classifier', help='Training models')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

# data preparation
with open('./data/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

positive_samples = all_data[:126]
unlabeled_samples = all_data[126:]


loader_p = DataLoader(positive_samples, batch_size=32, shuffle=True)
loader_u = DataLoader(unlabeled_samples, batch_size=32, shuffle=True)

model = PU_Classifier(num_node_features=all_data[0].n_node_features, num_edge_features=all_data[0].n_edge_features, 
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
prior = 0.001  # 正样本比例

train_loss = []
for epoch in range(args.epoch):
    model.train()
    total_loss = 0.0
    for (batch_p, batch_u) in zip(loader_p, loader_u):
        batch_p = batch_p.to(device)
        batch_u = batch_u.to(device)

        g_p = model(batch_p.x, batch_p.edge_index, batch_p.edge_attr, batch_p.batch)
        g_u = model(batch_u.x, batch_u.edge_index, batch_u.edge_attr, batch_u.batch)

        # g_p = model(batch_p.x, batch_p.edge_index, batch_p.batch)
        # g_u = model(batch_u.x, batch_u.edge_index, batch_u.batch)
        loss = nnPU_loss(g_p, g_u, prior)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_loss.append(total_loss / len(loader_p))
    print(f"Epoch {epoch+1:03d} | Loss = {total_loss/len(loader_p):.4f}")

plot_train_loss(args.epoch, train_loss, args.models)
