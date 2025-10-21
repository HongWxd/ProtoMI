import torch
import argparse
import pickle
from torch_geometric.loader import DataLoader
from model import GCN, GINE
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from utils.data_loader_CL import ContrastiveGraphDataset, contrastive_collate_fn


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

# load the graph data and descriptors data
with open('./data/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

positive_samples = all_data[:126]
unlabeled_samples = all_data[126:]
pos_loader = DataLoader(positive_samples, batch_size=128, shuffle=False)
unl_loader = DataLoader(unlabeled_samples, batch_size=128, shuffle=False)

if args.models == 'GINE':
    encoder = GINE(num_node_features=all_data[0].n_node_features, num_edge_features=all_data[0].n_edge_features, 
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
elif args.models == 'GCN':
    encoder = GCN(num_node_features=all_data[0].n_node_features, num_edge_features=all_data[0].n_edge_features, 
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
encoder.load_state_dict(torch.load(f'./checkpoints/CL_encoder_{args.models}.pth')) # load the checkpoints
print(f'Encoder {args.models} is loaded!')

encoder.eval()
def get_embeddings(encoder, dataloader):
    all_z = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if args.models == 'GCN':
                z = encoder(batch.x, batch.edge_index, batch.batch)
            elif args.models == 'GINE':
                z = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)

emb_pos = get_embeddings(encoder, pos_loader)
emb_unlab = get_embeddings(encoder, unl_loader)

# 计算相似度
sim_matrix = torch.mm(F.normalize(emb_unlab, dim=1), F.normalize(emb_pos, dim=1).T)
mean_sim = sim_matrix.mean(dim=1)
print(mean_sim[:10])

# 排序
sorted_idx = torch.argsort(mean_sim, descending=True)
ranked_unlabeled = [unlabeled_samples[i].id for i in sorted_idx]
searching_space_df = pd.read_csv('./data/searching_space_data.csv')
for id in ranked_unlabeled[:10]:
    formula = str(searching_space_df.loc[searching_space_df['cid'] == float(id), 'formula'].values[0])
    print(f'CID: {id}, Formula: {formula}')
