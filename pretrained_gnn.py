import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from model import GCN, GINE, GINE_descriptor, ProjectionHead
from tqdm import tqdm
import time
import pickle
from utils.tools import plot_train_results, self_training
from utils.data_loader_CL import ContrastiveGraphDataset, contrastive_collate_fn
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
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
parser.add_argument('--epoch', type=int, default=3000, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--folds', type=int, default=10, help='Fold number of cross validation')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
parser.add_argument('--training_methods', type=str, default='Dummy', help='Training methods')
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold of self training')
parser.add_argument('--warm_up_epoch', type=int, default=30, help='Self training warm up epoch period')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension of attention')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
parser.add_argument('--desp_dim', type=int, default=217, help='Number of descriptors')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


def info_nce_loss(z1, z2, temperature=0.5):
    """
    z1, z2: [N, D] - torch tensors
    returns scalar loss
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim_matrix = torch.matmul(representations, representations.T) / temperature  # [2N,2N]

    # mask out self-similarity
    diag_mask = torch.eye(2 * N, device=sim_matrix.device).bool()
    sim_matrix.masked_fill_(diag_mask, -9e15)

    # positive index: for i in [0..N-1], pos is i+N; for i in [N..2N-1], pos is i-N
    pos_indices = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(sim_matrix.device)

    log_prob = F.log_softmax(sim_matrix, dim=1)  # along columns (sum over negatives+pos)
    loss = -log_prob[torch.arange(2 * N), pos_indices].mean()
    return loss


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

encoder = GCN(num_node_features=all_data[0].n_node_features, num_edge_features=all_data[0].n_edge_features, 
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
projector = ProjectionHead(in_dim=args.embed_dim, proj_dim=128).to(device)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=args.learning_rate)

for epoch in range(args.epoch):  
    encoder.train()
    total_loss = 0

    for i, (pairs, label) in enumerate(loader):
        print(len(pairs), label.shape)

        g1_list = [p[0] for p in pairs]
        g2_list = [p[1] for p in pairs]

        batch1 = Batch.from_data_list(g1_list).to(device)
        batch2 = Batch.from_data_list(g2_list).to(device)

        h1 = encoder(batch1.x, batch1.edge_index, batch1.batch)
        h2 = encoder(batch2.x, batch2.edge_index, batch2.batch)

        z1 = projector(h1)
        z2 = projector(h2)

        loss = info_nce_loss(z1, z2, temperature=0.5)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        iter_loss = loss.item()
        print(f'\t  Iteras: {i+1} | Loss: {iter_loss:.7f}')

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")



# if best_model_state_dict is not None:
#     torch.save(best_model_state_dict, './checkpoints/best_model.pth')
