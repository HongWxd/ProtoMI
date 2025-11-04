import os
import torch
import argparse
import pandas as pd
from torch_geometric.loader import DataLoader
from model import GCN, GINE, ProjectionHead, Cluster_GINE
from tqdm import tqdm
import time
import umap
import pickle
from utils.tools import plot_train_loss, perturb_edges, info_nce_loss, tanimoto_matrix
from utils.graph_augmentation import Graph_Augmentation_Helper
from sklearn.model_selection import KFold
import numpy as np
import torch.nn.functional as F
import warnings
from torch_geometric.data import Batch

from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from rdkit import DataStructs


warnings.filterwarnings('ignore')

# unsupervised learning configs
parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=1500, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--training_types', type=str, default='Unsupervised learning', help='training_types')
parser.add_argument('--models', type=str, default='GINE', help='Training models')
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


def unsupervised_training(pos_train_samples, pos_test_samples):
    # Unsupervised training
    # train a GNN model to represent all positive training data and get the prototypes
    pos_train_samples = pos_train_samples + pos_test_samples
    train_loader = DataLoader(pos_train_samples, batch_size=args.batch_size, shuffle=True)
    model = Cluster_GINE(num_node_features=pos_train_samples[0].n_node_features, num_edge_features=pos_train_samples[0].n_edge_features, 
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
    projection_head1 = ProjectionHead(in_dim=args.hidden_channels).to(device)
    projection_head2 = ProjectionHead(in_dim=args.hidden_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    model.train()
    unsuper_train_loss = []
    for epoch in tqdm(range(1, args.epoch + 1), desc='Training the representation GNN...'):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)

            # graph augmentation: for constractive learning
            data_aug1 = data.clone()
            data_aug2 = perturb_edges(data.clone(), device)

            out1 = model(data_aug1.x, data_aug1.edge_index, data_aug1.edge_attr, data_aug1.batch)
            out2 = model(data_aug2.x, data_aug2.edge_index, data_aug2.edge_attr, data_aug2.batch)
            pro_out1 = projection_head1(out1)
            pro_out2 = projection_head2(out2)

            loss = info_nce_loss(pro_out1, pro_out2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        unsuper_train_loss.append(avg_loss)
        print(f"Epoch [{epoch}/{args.epoch}]  Loss: {avg_loss}")
    plot_train_loss(args.epoch, unsuper_train_loss, args.models, args.training_types)
    torch.save(model.state_dict(), f'./checkpoints/{args.training_types}_model_{args.models}.pth')

    return model


def load_data(data_path):
    # data preparation
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)

    positive_samples = all_data[:126] # number of positive samples
    unlabeled_samples = all_data[126:]
    graph_aug_helper = Graph_Augmentation_Helper(positive_samples, args)
    pos_train_samples, pos_test_samples = graph_aug_helper.train_test_split_positive_samples()
    return positive_samples, unlabeled_samples, pos_train_samples, pos_test_samples


def get_representation_model(file_path, pos_train_samples, pos_test_samples):
    if os.path.exists(file_path):
            model = Cluster_GINE(num_node_features=pos_train_samples[0].n_node_features, num_edge_features=pos_train_samples[0].n_edge_features, 
                hidden_channels=args.hidden_channels,
                num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
            model.load_state_dict(torch.load(file_path)) # load the checkpoints
    else:
        model = unsupervised_training(pos_train_samples, pos_test_samples)

    return model


def main():
    data_path = './data/all_data.pkl'
    file_path = f"./checkpoints/{args.training_types}_model_{args.models}.pth"

    # load data
    positive_samples, unlabeled_samples, pos_train_samples, pos_test_samples = load_data(data_path)
    pos_aug_samples = pos_train_samples + pos_test_samples

    # check and get the representation model checkpoint
    model = get_representation_model(file_path, pos_train_samples, pos_test_samples)

    # get the original cluster in the positive samples
    model.eval()
    projection_head = ProjectionHead(in_dim=args.hidden_channels).to(device)
    pos_aug_loader = DataLoader(pos_aug_samples, batch_size=args.batch_size, shuffle=False)

    all_embeddings = []
    with torch.no_grad():
        for data in pos_aug_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            emb = projection_head(out)
            all_embeddings.append(emb.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()  # shape: [num_molecules, hidden_dim]
    print(all_embeddings.shape)

    sim_matrix = tanimoto_matrix(all_embeddings)
    # 转化为距离矩阵用于降维
    dist_matrix = 1 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)  # 对角线必须为 0

    reducer_2d = umap.UMAP(random_state=42)
    embeddings = reducer_2d.fit_transform(dist_matrix)

    # 2️⃣ 进行层次聚类 (ward/linkage 可换成 'average'、'complete')
    Z = linkage(dist_matrix, method='ward')
    additives_names = range(len(all_embeddings))
    # 3️⃣ 画树状图
    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=additives_names, leaf_rotation=90)
    plt.title("Hierarchical Clustering of Additives (126 positive samples)")
    plt.xlabel("Additive")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("./V3/plots/additives_hierarchical_clustering.png", dpi=600)


    possible_clusters = range(2, 11)  # 尝试从2到10个簇
    best_score = -1
    best_k = None
    best_labels = None

    for k in possible_clusters:
        cluster_labels = fcluster(Z, t=k, criterion='maxclust')
        try:
            score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
            print(f"k={k}, silhouette score={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = cluster_labels 
        except Exception as e:
            print(f"k={k} failed: {e}")

    print(f"\n✅ 最佳簇数: {best_k}, 对应的平均轮廓系数: {best_score:.4f}")

    # 可视化不同簇在UMAP上的分布
    plt.figure(figsize=(8,6))
    for i in range(1, best_k+1):
        plt.scatter(all_embeddings[best_labels==i, 0], all_embeddings[best_labels==i, 1], s=40, label=f"Cluster {i}", alpha=0.7)
    plt.legend()
    plt.title(f"UMAP Projection (Best Clusters = {best_k})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig("./V3/plots/additives_umap_best_cluster.png", dpi=600)


if __name__=="__main__":
    main()

        


