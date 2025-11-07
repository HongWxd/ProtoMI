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
from utils.tools import plot_train_loss, perturb_edges, info_nce_loss, try_multiple_cluster_combinations
from utils.graph_augmentation import Graph_Augmentation_Helper
from utils.visualization import show_gnn_fp_consistency_results, plot_hierarchical_cluster_dendrogram, plot_cluster_distribution_UMAP
from sklearn.model_selection import KFold
import numpy as np
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from random import sample


warnings.filterwarnings('ignore')

# unsupervised learning configs
parser = argparse.ArgumentParser(description="Train the model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
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

# prototypes configs
parser.add_argument('--max_cluster', type=int, default=10, help='max cluster number')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature coefficient for prototypes')
parser.add_argument('--proto_epoch', type=int, default=500, help='Number of training epochs')
parser.add_argument('--r', type=int, default=1000, help='number of randomly select neg prototypes')
parser.add_argument('--proto_training_types', type=str, default='Prototype contrastive learning', help='training_types')
parser.add_argument('--proto_models', type=str, default='GINE', help='model name for PCL')



# main configs
parser.add_argument('--task', type=str, default='train', help='task types')

args = parser.parse_args()
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')


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
    unl_train_samples, unl_test_samples = train_test_split(unlabeled_samples, test_size=args.test_size, random_state=args.random_state)
    return positive_samples, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples


def get_representation_model(file_path, pos_train_samples, pos_test_samples):
    if os.path.exists(file_path):
            model = Cluster_GINE(num_node_features=pos_train_samples[0].n_node_features, num_edge_features=pos_train_samples[0].n_edge_features, 
                hidden_channels=args.hidden_channels,
                num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
            model.load_state_dict(torch.load(file_path)) # load the checkpoints
    else:
        model = unsupervised_training(pos_train_samples, pos_test_samples)

    return model


def get_prototypes(model, pos_samples, unlabeled_samples):
    # get the original cluster in the positive samples
    model.eval()
    projection_head = ProjectionHead(in_dim=args.hidden_channels).to(device)
    pos_sample_loader = DataLoader(pos_samples, batch_size=args.batch_size, shuffle=False)
    unl_sample_loader = DataLoader(unlabeled_samples, batch_size=args.batch_size, shuffle=False)
    pos_graph_embeddings = []
    pos_additives_names = []
    unl_graph_embeddings = []
    unl_additives_names = []
    with torch.no_grad():
        for data in pos_sample_loader:
            pos_additives_names += data.id
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            emb = projection_head(out)
            pos_graph_embeddings.append(emb.cpu())
            
        
        for data in unl_sample_loader:
            unl_additives_names += data.id
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            emb = projection_head(out)
            unl_graph_embeddings.append(emb.cpu())
            

    pos_graph_embeddings = torch.cat(pos_graph_embeddings, dim=0).numpy()  # shape: [num_molecules, hidden_dim]
    unl_graph_embeddings = torch.cat(unl_graph_embeddings, dim=0).numpy()

    reducer_2d = umap.UMAP(random_state=42)
    umap_embeddings = reducer_2d.fit_transform(pos_graph_embeddings)

    # hierarchical cluster
    Z = linkage(pos_graph_embeddings, method='ward', metric='euclidean')

    # get the best cluster number of all positive samples
    best_cluster_num, labels = try_multiple_cluster_combinations(Z, pos_graph_embeddings, args)
    
    # plot hierarchical cluster dendrogram
    plot_hierarchical_cluster_dendrogram(Z, pos_additives_names)
    # plot UMAP cluster distribution
    plot_cluster_distribution_UMAP(best_cluster_num, labels, umap_embeddings)
    # show the consistency analysis results
    if args.task == 'eval':
        show_gnn_fp_consistency_results(pos_additives_names, umap_embeddings)

    # get the positive samples prototypes
    proto_centroids = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = pos_graph_embeddings[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        proto_centroids.append(centroid)
    
    proto_centroids = np.array(proto_centroids)
    
    dist_matrix = cdist(unl_graph_embeddings, proto_centroids, metric='euclidean')  # shape: (180000, 10)
    closest_proto = np.argmin(dist_matrix, axis=1)
    closest_proto += 1

    # get unlabeled samples prototypes
    all_graph_embeddings = np.vstack([pos_graph_embeddings, unl_graph_embeddings])
    all_prototypes = np.concatenate([np.array(labels), closest_proto])
    densities = []
    for label in unique_labels:
        select_embeddings = all_graph_embeddings[all_prototypes == label]
        centroid = proto_centroids[unique_labels == label]

        distances = cdist(select_embeddings, centroid.reshape(1, -1), metric='euclidean')
        d = (np.sqrt(distances).mean()) / np.log(len(select_embeddings) + 10)
        densities.append(d)
    
    # clamp extreme values for stability
    low, high = np.percentile(densities, 10), np.percentile(densities, 90)
    densities = np.clip(densities, low, high)

    # === scale the mean to temperature  ===
    densities = args.temperature * densities / densities.mean()
    densities = np.array(densities)
    samples_densities = np.array([densities[unique_labels == label].item() for label in all_prototypes])
    
    pos_additives_ids = [int(i.item()) for i in pos_additives_names]
    unl_additives_ids = [int(i.item()) for i in unl_additives_names]
    molecules_id = pos_additives_ids + unl_additives_ids

    prototypes_table = pd.DataFrame()
    prototypes_table['molecule_id'] = molecules_id
    prototypes_table['prototypes'] = all_prototypes
    prototypes_table['density'] = samples_densities

    # get the prototype embeddings for each sample
    prototypes_emb = proto_centroids[np.array(all_prototypes) - 1]

    return prototypes_table, prototypes_emb


def prototype_contrastive_training():
    a=1


def main():
    data_path = './data/all_data.pkl'
    file_path = f"./checkpoints/{args.training_types}_model_{args.models}.pth"

    # load data
    positive_samples, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples = load_data(data_path)
    all_pos_samples = pos_train_samples + pos_test_samples

    # check and get the representation model checkpoint
    model = get_representation_model(file_path, pos_train_samples, pos_test_samples)

    # get the prototypes table
    if args.task == 'train':
        prototypes_table, prototype_embeddings = get_prototypes(model, pos_train_samples, unl_train_samples) # get the prototypes during training process
    elif args.task == 'test':
        prototypes_table, prototype_embeddings = get_prototypes(model, pos_test_samples, unl_test_samples) # get the prototypes during test process
    elif args.task == 'eval':
        prototypes_table, prototype_embeddings = get_prototypes(model, positive_samples, unlabeled_samples) # get the prototypes of all positive samples to analysis the model performance

    # train the prototype contrastive learning model
    proto_train_samples = pos_train_samples + unl_train_samples
    proto_test_samples = pos_test_samples + unl_test_samples
    proto_train_loader = DataLoader(proto_train_samples, batch_size=args.batch_size, shuffle=True)
    proto_test_loader = DataLoader(proto_test_samples, batch_size=args.batch_size, shuffle=False)
    
    molecule_id = prototypes_table['molecule_id'].values.tolist()
    prototypes = prototypes_table['prototypes'].values.tolist()
    densities = prototypes_table['density'].values.tolist()
    molecule_id = torch.tensor(molecule_id, dtype=torch.int)
    prototype_embeddings = torch.tensor(prototype_embeddings, dtype=torch.float)
    densities = torch.tensor(densities, dtype=torch.float)

    encoder = GINE(num_node_features=proto_train_samples[0].n_node_features, num_edge_features=proto_train_samples[0].n_edge_features, 
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
    projection = ProjectionHead(in_dim=args.hidden_channels).to(device)
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    encoder.train()
    proto_train_loss = []
    for epoch in tqdm(range(1, args.proto_epoch + 1), desc='Training the prototype contrastive learning model...'):
        total_loss = 0
        for i, data in enumerate(proto_train_loader):
            molecule_id_list = molecule_id.tolist()
            data_id_list = data.id.tolist()
            # neg_proto_id_all = list(set(molecule_id_list) - set(data_id_list))
            neg_proto_id_all = [x for x in molecule_id_list if x not in data_id_list]
            # print(len(neg_proto_id_all), len(set(molecule_id_list)), len(set(data_id_list)))

            data = data.to(device)
            molecule_id = molecule_id.to(device)
            prototype_embeddings = prototype_embeddings.to(device)
            densities = densities.to(device)
        
            pos_mask = torch.isin(molecule_id, data.id)
            pos_prototypes = prototype_embeddings[pos_mask]
            
            neg_proto_id = torch.tensor(sample(neg_proto_id_all, args.r), dtype=torch.int).to(device)
            neg_mask = torch.isin(molecule_id, neg_proto_id)
            # print(len(neg_proto_id), neg_mask.sum())
            neg_prototypes = prototype_embeddings[neg_mask]

            proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)

            query = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            query = projection(query)

            logits_proto = torch.mm(query, proto_selected.t())
            labels_proto = torch.linspace(0, query.size(0)-1, steps=query.size(0)).long().to(device)

            # scaling temperatures for the selected prototypes
            selected_mask = pos_mask | neg_mask
            temp_proto = densities[selected_mask]
            logits_proto /= temp_proto

            loss = criterion(logits_proto, labels_proto)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iter_loss = loss / (pos_mask.sum() + neg_mask.sum())
            # print(f'\t  Iteras: {i+1} | Loss: {iter_loss:.7f}')

        avg_loss = total_loss / len(molecule_id.tolist())
        proto_train_loss.append(avg_loss)
        print(f"Epoch [{epoch}/{args.proto_epoch}]  Loss: {avg_loss}")
    plot_train_loss(args.epoch, proto_train_loss, args.proto_models, args.proto_training_types)




if __name__=="__main__":
    main()

        


