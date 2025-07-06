import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.manifold import TSNE
import torch
import argparse
import pickle
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU, MultiheadAttention, LayerNorm
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool
import umap
from sklearn.preprocessing import StandardScaler


class DANLayer(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, d_keys, d_values, d_ff, dropout):
        super(DANLayer, self).__init__()

        self.query_projection = Linear(hidden_channels, hidden_channels)
        self.key_projection = Linear(hidden_channels, hidden_channels)
        self.value_projection = Linear(hidden_channels, hidden_channels)

        self.multiheadattn = MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.attn_norm = LayerNorm(hidden_channels)

        self.lin1 = Linear(hidden_channels, d_ff)
        self.relu = ReLU()
        self.lin2 = Linear(d_ff, hidden_channels)
        self.dropout = Dropout(dropout)
        self.ffn_attn_norm = LayerNorm(hidden_channels)
    
    def feedforward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        return x

    def forward(self, x):
        q = self.query_projection(x)
        k = self.query_projection(x)
        v = self.value_projection(x)
        x_in = x

        x, _ = self.multiheadattn(q, k, v)
        x = x_in + self.attn_norm(x) # [B, 1, d_keys * num_heads]

        x_ffn = x
        x = self.feedforward(x)
        x = x_ffn + self.ffn_attn_norm(x)

        return x

class DAN(torch.nn.Module): # Descriptor Attention Network
    def __init__(self, hidden_channels, num_heads, desp_dim, d_ff, dropout, d_keys, d_values):
        super(DAN, self).__init__()

        self.desp_num = desp_dim

        self.multihead_attn = MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.desp_embed = Linear(desp_dim, hidden_channels * desp_dim)
        self.desp_num = desp_dim
        self.attn_norm = LayerNorm(hidden_channels)

        self.lin1 = Linear(hidden_channels, d_ff)
        self.relu = ReLU()
        self.lin2 = Linear(d_ff, hidden_channels)
        self.dropout = Dropout(dropout)
        self.ffn_attn_norm = LayerNorm(hidden_channels)

        self.dan_layer1 = DANLayer(hidden_channels, num_heads, d_keys, d_values, d_ff, dropout)
        self.dan_layer2 = DANLayer(hidden_channels, num_heads, d_keys, d_values, d_ff, dropout)
        self.dan_layer3 = DANLayer(hidden_channels, num_heads, d_keys, d_values, d_ff, dropout)
    
    def feedforward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        return x

    def forward(self, x, descriptors):
        B = x.shape[0]
        H = x.shape[1]
        N = self.desp_num
        
        desp_embed = self.desp_embed(descriptors) 
        desp_embed = desp_embed.view(B, N, H) # [batchsize, num_desp_features * hidden_channels] --> [batchsize, num_desp_features, hidden_channels]
        x_in = x.unsqueeze(1)  # [B, 1, hidden_channels]

        x, _ = self.multihead_attn(x_in, desp_embed, desp_embed)
        x = x_in + self.attn_norm(x) # [B, 1, H]

        x_ffn = x
        x = self.feedforward(x)
        x = x_ffn + self.ffn_attn_norm(x)

        x = self.dan_layer1(x)

        return x

class GINE_descriptor(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes, dropout, args):
        super(GINE_descriptor, self).__init__()

        nn1 = Sequential(Linear(num_node_features, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn1, edge_dim=num_edge_features)

        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv2 = GINEConv(nn2, edge_dim=num_edge_features)

        nn3 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv3 = GINEConv(nn3, edge_dim=num_edge_features)

        self.multihead_attn = MultiheadAttention(args.hidden_channels, args.num_heads, batch_first=True)
        self.desp_embed = Linear(args.desp_dim, args.hidden_channels * args.desp_dim)
        self.desp_num = args.desp_dim
        self.attn_norm = LayerNorm(args.hidden_channels)

        self.ffn_lin1 = Linear(hidden_channels, args.d_ff)
        self.relu = ReLU()
        self.ffn_lin2 = Linear(args.d_ff, hidden_channels)
        self.ffn_dropout = Dropout(dropout)
        self.ffn_attn_norm = LayerNorm(hidden_channels)

        self.dan1 = DAN(hidden_channels, args.num_heads, args.desp_dim, args.d_ff, dropout, args.d_keys, args.d_values)
        
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(dropout)

    def feedforward(self, x):
        x = self.ffn_lin1(x)
        x = self.relu(x)
        x = self.ffn_dropout(x)
        x = self.ffn_lin2(x)

        return x

    def forward(self, x, edge_index, edge_attr, batch, descriptors):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)# [batchsize, hidden_channels]

        x = self.dan1(x, descriptors)
        # desp_out, _ = self.multihead_attn(desp_embed, x_in, x_in) # [B, N, H]
        x = x.squeeze(1)

        self.embeds = x

        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--d_ff', type=int, default=512, help='Number of model dimension')
parser.add_argument('--epoch', type=int, default=300, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--folds', type=int, default=10, help='Fold number of cross validation')
parser.add_argument('--repeats', type=int, default=5, help='Repeat number of cross validation')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
parser.add_argument('--training_methods', type=str, default='Self_Training', help='Training methods')
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold of self training')
parser.add_argument('--warm_up_epoch', type=int, default=40, help='Self training warm up epoch period')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension of attention')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
parser.add_argument('--desp_dim', type=int, default=217, help='Number of descriptors')
parser.add_argument('--d_keys', type=int, default=128, help='Number of descriptors')
parser.add_argument('--d_values', type=int, default=128, help='Number of descriptors')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

def plot_kmeans_clusters(all_embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kmeans = kmeans.fit_predict(all_embeddings)
    kmeans_result = labels_kmeans
    reducer = umap.UMAP(random_state=42)
    emb_2d = reducer.fit_transform(all_embeddings)

    # reducer = umap.UMAP(n_components=3, random_state=42)
    # emb_3d = reducer.fit_transform(all_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(emb_2d[:,0], emb_2d[:,1], c=kmeans_result, cmap='tab10', s=10)

    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=kmeans_result, cmap='tab10', s=10)
    plt.title('UMAP projection with KMeans clusters')
    plt.tight_layout()
    plt.savefig('./figs/KMeans_clusters.png')

def plot_dbscan_clusters(all_embeddings):
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_dbscan = dbscan.fit_predict(all_embeddings)
    dbscan_result = labels_dbscan
    reducer = umap.UMAP(random_state=42)
    emb_2d = reducer.fit_transform(all_embeddings)

    plt.figure(figsize=(8,6))
    plt.scatter(emb_2d[:,0], emb_2d[:,1], c=dbscan_result, cmap='tab10', s=10)
    plt.title('UMAP projection with DBSCAN clusters')
    plt.tight_layout()
    plt.savefig('./figs/DBSCAN_clusters.png')

def plot_meanshift_clusters(all_embeddings, n_clusters):
    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agglo = agglo.fit_predict(all_embeddings)
    agglo_results = labels_agglo
    reducer = umap.UMAP(random_state=42)
    emb_2d = reducer.fit_transform(all_embeddings)

    plt.figure(figsize=(8,6))
    plt.scatter(emb_2d[:,0], emb_2d[:,1], c=agglo_results, cmap='tab10', s=10)
    plt.title('UMAP projection with Agglomerative Clustering')
    plt.tight_layout()
    plt.savefig('./figs/Agglomerative_clusters.png')

# load the graph data and descriptors data
with open('./data/norm_normal.pkl', 'rb') as f:
    desp_data = pickle.load(f)
with open('./data/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

merged_data = []
desp_data = torch.tensor(desp_data, dtype=torch.float)
for desp, graph in zip(tqdm(desp_data, desc='Loading training data...'), all_data):
    graph.descriptors = desp.unsqueeze(0)
    merged_data.append(graph)
all_data = merged_data

pred_data = pd.DataFrame(pd.read_csv('data/predict_1.csv'))
selected_cids = pred_data['cid'].tolist()

selected_data = []
for graph in all_data:
    if graph.cid in selected_cids:
        selected_data.append(graph)
selected_loader = DataLoader(selected_data, batch_size=args.batch_size, shuffle=False)

model = GINE_descriptor(num_node_features=selected_data[0].n_node_features, num_edge_features=selected_data[0].n_edge_features, 
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
model.load_state_dict(torch.load('./checkpoints/best_model.pth')) # load the checkpoints
print('Model is loaded!')

model.eval()
all_embeddings = []
with torch.no_grad():
    all_preds = {}
    for data in tqdm(selected_loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.descriptors)
        embeds = model.embeds.cpu()
        all_embeddings.append(embeds)
all_embeddings = all_embeddings[:-1]
all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
print(all_embeddings.shape)
n_clusters = 6

scaler = StandardScaler()
all_embeddings_scaled = scaler.fit_transform(all_embeddings)

plot_kmeans_clusters(all_embeddings_scaled, n_clusters)
# plot_dbscan_clusters(all_embeddings_scaled)
# plot_meanshift_clusters(all_embeddings_scaled, n_clusters)
