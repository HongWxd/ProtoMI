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
parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--folds', type=int, default=10, help='Fold number of cross validation')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
parser.add_argument('--models', type=str, default='GINE', help='Training models')
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold of self training')
parser.add_argument('--warm_up_epoch', type=int, default=30, help='Self training warm up epoch period')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension of attention')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
parser.add_argument('--desp_dim', type=int, default=217, help='Number of descriptors')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


def contrastive_loss(z1, z2, labels, margin=1.0):
    """
    z1, z2: [N, D]
    labels: [N] (1 for positive, 0 for negative)
    """
    distances = F.pairwise_distance(z1, z2, keepdim=True)  # [N, 1]
    
    loss_pos = labels * distances.pow(2)
    loss_neg = (1 - labels) * F.relu(margin - distances).pow(2)
    loss = 0.5 * (loss_pos + loss_neg)
    
    return loss.mean()


# data preparation
with open('./data/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

positive_samples = all_data[:126]
unlabeled_samples = all_data[126:]
dataset = ContrastiveGraphDataset(positive_samples, unlabeled_samples, ratio=5)
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    collate_fn=lambda b: contrastive_collate_fn(b, dataset)
) 


# model preparation
if args.models == 'GCN':
    encoder = GCN(num_node_features=all_data[0].n_node_features, num_edge_features=all_data[0].n_edge_features, 
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
elif args.models == 'GINE':
    encoder = GINE(num_node_features=all_data[0].n_node_features, num_edge_features=all_data[0].n_edge_features, 
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)

projector = ProjectionHead(in_dim=args.embed_dim, proj_dim=128).to(device)


# model training
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=args.learning_rate)

train_loss = []
for epoch in range(args.epoch):  
    encoder.train()
    total_loss = 0

    for i, (pairs, label) in enumerate(loader):
        g1_list = [p[0] for p in pairs]
        g2_list = [p[1] for p in pairs]

        batch1 = Batch.from_data_list(g1_list).to(device)
        batch2 = Batch.from_data_list(g2_list).to(device)
        label = label.to(device)

        if args.models == 'GCN':
            # GCN
            h1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
            h2 = encoder(batch2.x, batch2.edge_index, batch2.batch)
        elif args.models == 'GINE':
            # GINE
            h1 = encoder(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
            h2 = encoder(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)

        z1 = projector(h1)
        z2 = projector(h2)

        loss = contrastive_loss(z1, z2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_loss.append(total_loss / len(pairs))
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(pairs)}")

plot_train_loss(args.epoch, train_loss, args.models)

# Save the model
torch.save(encoder.state_dict(), f'./checkpoints/CL_encoder_{args.models}.pth')
