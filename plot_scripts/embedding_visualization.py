import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import argparse
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle
import numpy as np
import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout, Sequential, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GINEConv
from sklearn.model_selection import train_test_split
import imageio.v2 as imageio
import os

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=800, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--folds', type=int, default=5, help='fold number of cross validation')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x

class GCN_with_edge_attr(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes, dropout):
        super(GCN_with_edge_attr, self).__init__()

        nn1 = Sequential(Linear(num_node_features, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn1, edge_dim=num_edge_features)

        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv2 = GINEConv(nn2, edge_dim=num_edge_features)

        self.lin = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x

def visualize_embeddings(model, dataloader, epoch):
    model.eval()
    all_embeds = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            embeds = model(data.x, data.edge_index, data.edge_attr, data.batch)
            all_embeds.append(embeds.cpu())
            all_labels.append(data.y.cpu())

    embeds = torch.cat(all_embeds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    reducer = PCA(n_components=2)
    embeds_2d = reducer.fit_transform(embeds)

    plt.figure(figsize=(6, 6))
    num_classes = len(np.unique(labels))

    for i in range(num_classes):
        idx = labels == i
        plt.scatter(embeds_2d[idx, 0], embeds_2d[idx, 1], label=f'Class {i}', alpha=0.7, s=20)

    plt.legend()
    plt.title(f'Graph Embedding at Epoch {epoch}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./figs/embedding_evol/embed_epoch_{epoch:03d}.png')
    plt.close()

with open('./data/labeled_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42, shuffle=True)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# model = GCN(num_node_features=train_data[0].n_node_features,
#         hidden_channels=args.hidden_channels,
#         num_classes=args.num_classes, dropout=args.dropout).to(device)

model = GCN_with_edge_attr(num_node_features=train_data[0].n_node_features, num_edge_features=train_data[0].n_edge_features, 
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes, dropout=args.dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

embeddings_over_time = []
for epoch in tqdm(range(1, args.epoch + 1), desc='Training'):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    visualize_embeddings(model, train_loader, epoch)

def make_gif(image_folder, output_path='./figs/embedding_evolution.gif'):
    images = []
    for epoch in sorted(os.listdir(image_folder)):
        if epoch.endswith('.png'):
            images.append(imageio.imread(os.path.join(image_folder, epoch)))
    imageio.mimsave(output_path, images, duration=0.4)

make_gif(image_folder='./figs/embedding_evol')