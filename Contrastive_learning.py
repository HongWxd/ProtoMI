import os
import torch
import argparse
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.nn import BCEWithLogitsLoss
from model import GCN, GINE, ProjectionHead, Cluster_GINE, ProjectionHead_PCL
from tqdm import tqdm
import copy
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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from random import sample
import seaborn as sns


warnings.filterwarnings('ignore')

# unsupervised learning configs
parser = argparse.ArgumentParser(description="Train the model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--usl_batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--usl_learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--usl_hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--training_types', type=str, default='Unsupervised learning', help='training_types')
parser.add_argument('--models', type=str, default='GINE', help='Training models')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension of attention')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
parser.add_argument('--desp_dim', type=int, default=217, help='Number of descriptors')
parser.add_argument('--retrain_usl', type=bool, default=False, help='retrain the usl models')
parser.add_argument('--ucl_trials', type=int, default=10, help='Number of trials for unsupervised learning')

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
parser.add_argument('--proto_epoch', type=int, default=20, help='Number of training epochs')
parser.add_argument('--r', type=int, default=10000, help='number of randomly select neg prototypes')
parser.add_argument('--proto_training_types', type=str, default='Prototype contrastive learning', help='training_types')
parser.add_argument('--proto_models', type=str, default='GINE', help='model name for PCL')
parser.add_argument('--pcl_hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--pcl_learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--pcl_batch_size', type=int, default=1024, help='Batch size for training')
parser.add_argument('--threshold', type=float, default=0.3, help='threshold')


# main configs
parser.add_argument('--task', type=str, default='train', help='task types')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


def unsupervised_training(pos_train_samples, pos_test_samples):
    # Unsupervised training
    # train a GNN model to represent all positive training data and get the prototypes
    pos_train_samples = pos_train_samples + pos_test_samples
    train_loader = DataLoader(pos_train_samples, batch_size=args.usl_batch_size, shuffle=True)
    model = Cluster_GINE(num_node_features=pos_train_samples[0].n_node_features, num_edge_features=pos_train_samples[0].n_edge_features, 
            hidden_channels=args.usl_hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
    projection_head1 = ProjectionHead(in_dim=args.usl_hidden_channels).to(device)
    projection_head2 = ProjectionHead(in_dim=args.usl_hidden_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.usl_learning_rate, weight_decay=5e-4)
    
    unsuper_train_loss = []
    silhouette_scores = 0
    best_model = None
    for epoch in tqdm(range(1, args.epoch + 1), desc='Training the representation GNN...'):
        model.train()
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

        model.eval()
        eval_loader = DataLoader(pos_train_samples, batch_size=args.usl_batch_size, shuffle=False)
        all_embeddings = []
        with torch.no_grad():
            for data in eval_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                all_embeddings.append(out.cpu())

        all_embeddings = F.normalize(torch.cat(all_embeddings), dim=-1).numpy()

        # hierarchical cluster
        Z = linkage(all_embeddings, method='average', metric='cosine')

        # get the best cluster number of all positive samples
        best_cluster_num, labels = try_multiple_cluster_combinations(Z, all_embeddings, args)
        sil = silhouette_score(all_embeddings, labels, metric='cosine')


        if sil > silhouette_scores:
            silhouette_scores = sil
            best_model = copy.deepcopy(model)
            print(f'Update! Epoch: {epoch}, silhouette score: {sil}')

        print(f"Epoch [{epoch}/{args.epoch}]  Loss: {avg_loss}")
    plot_train_loss(args.epoch, unsuper_train_loss, args.models, args.training_types)

    return best_model, silhouette_scores


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
    if os.path.exists(file_path) and args.retrain_usl == False:
        model = Cluster_GINE(num_node_features=pos_train_samples[0].n_node_features, num_edge_features=pos_train_samples[0].n_edge_features, 
            hidden_channels=args.usl_hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
        model.load_state_dict(torch.load(file_path)) # load the checkpoints
        best_model = model
    else:
        best_model = None
        best_sil_score = -1
        for trial in tqdm(range(args.ucl_trials), desc=f'Unsupervised learning trials...'):
            model, silhouette_scores = unsupervised_training(pos_train_samples, pos_test_samples)
            if silhouette_scores > best_sil_score:
                best_sil_score = silhouette_scores
                best_model = copy.deepcopy(model)
                best_trial = trial + 1
        
        print(f'Best trial from unsupervised learning: {best_trial}')
        print(f'Best silhouette score from unsupervised learning: {best_sil_score}')

        torch.save(best_model.state_dict(), f'./checkpoints/{args.training_types}_model_{args.models}.pth')

    return best_model

def get_prototypes(model, pos_samples, unlabeled_samples):
    # get the original cluster in the positive samples
    model.eval()
    projection_head = ProjectionHead(in_dim=args.usl_hidden_channels).to(device)
    pos_sample_loader = DataLoader(pos_samples, batch_size=args.usl_batch_size, shuffle=False)
    pos_graph_embeddings = []
    pos_additives_names = []

    with torch.no_grad():
        for data in pos_sample_loader:
            pos_additives_names += data.id
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            emb = projection_head(out)
            pos_graph_embeddings.append(emb.cpu())
            

    pos_graph_embeddings = F.normalize(torch.cat(pos_graph_embeddings), dim=-1).numpy()  # shape: [num_molecules, hidden_dim]

    reducer_2d = umap.UMAP(random_state=42)
    umap_embeddings = reducer_2d.fit_transform(pos_graph_embeddings)

    # hierarchical cluster
    Z = linkage(pos_graph_embeddings, method='average', metric='cosine')

    # get the best cluster number of all positive samples
    best_cluster_num, labels = try_multiple_cluster_combinations(Z, pos_graph_embeddings, args)
    
    # plot hierarchical cluster dendrogram
    plot_hierarchical_cluster_dendrogram(Z, pos_additives_names)
    # plot UMAP cluster distribution
    plot_cluster_distribution_UMAP(best_cluster_num, labels, umap_embeddings)
    # show the consistency analysis results
    if args.task == 'eval':
        show_gnn_fp_consistency_results(pos_additives_names, umap_embeddings)

    # # get the positive samples prototypes
    # proto_centroids = [] # [N, D]
    # unique_labels = np.unique(labels) # 1-N prototypes
    # for label in unique_labels:
    #     cluster_points = pos_graph_embeddings[labels == label]
    #     centroid = np.mean(cluster_points, axis=0)
    #     proto_centroids.append(centroid)
    
    # proto_centroids = np.array(proto_centroids)
    
    # dist_matrix = cdist(unl_graph_embeddings, proto_centroids, metric='cosine')  # shape: (180000, 10)
    # closest_proto = np.argmin(dist_matrix, axis=1)
    # closest_proto += 1

    # # get unlabeled samples prototypes
    # all_graph_embeddings = np.vstack([pos_graph_embeddings, unl_graph_embeddings]) # [pos + neg, D]
    # all_prototypes = np.concatenate([np.array(labels), closest_proto])
    # densities = [] # [N, ]
    # for label in unique_labels:
    #     select_embeddings = all_graph_embeddings[all_prototypes == label]
    #     centroid = proto_centroids[unique_labels == label]

    #     distances = cdist(select_embeddings, centroid.reshape(1, -1), metric='cosine')
    #     d = (np.sqrt(distances).mean()) / np.log(len(select_embeddings) + 10)
    #     densities.append(d)
    
    # # clamp extreme values for stability
    # low, high = np.percentile(densities, 10), np.percentile(densities, 90)
    # densities = np.clip(densities, low, high)

    # # === scale the mean to temperature  ===
    # densities = args.temperature * densities / densities.mean()
    # densities = np.array(densities)

    # samples_densities = []
    # for label in all_prototypes:
    #     sample_density = densities[unique_labels == label][0]
    #     samples_densities.append(sample_density)
    
    pos_additives_ids = [int(i.item()) for i in pos_additives_names]
    # unl_additives_ids = [int(i.item()) for i in unl_additives_names]
    # molecules_id = pos_additives_ids + unl_additives_ids

    prototypes_table = pd.DataFrame()
    prototypes_table['molecule_id'] = pos_additives_ids
    prototypes_table['prototypes'] = labels
    # prototypes_table['density'] = samples_densities
    prototypes_table.to_csv('./proto_table.csv', index=False)

    # get the prototype embeddings for each sample
    # prototypes_emb = proto_centroids[np.array(all_prototypes) - 1]
    # unl_proto_embeddings = proto_centroids[closest_proto - 1]
    # pos_proto_embeddings = proto_centroids[labels - 1]
    return prototypes_table


def prototype_contrastive_training(epoch, encoder, projection, optimizer, proto_train_loader, molecule_id, proto_label, pos_loader):
    id2idx = {pid.item(): idx for idx, pid in enumerate(molecule_id)}
    
    encoder.train()
    epoch_train_loss = 0
    total_samples = 0
    threshold = 0
    for i, data in enumerate(proto_train_loader):
        # move the data into cuda
        data = data.to(device)

        query = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        query = projection(query)
        query = F.normalize(query, dim=-1)
        

        # prototype regularization
        proto_sim = proto_centroids @ proto_centroids.t()
        decor_loss = ((proto_sim - torch.eye(proto_sim.size(0), device=proto_sim.device)) ** 2).mean()

        # compute similarities between query and prototypes
        sims = query @ proto_centroids.t()  # shape [batch_size, num_prototypes]
        max_sims, pos_idx = sims.max(dim=1)

        if epoch > 1:
            threshold = args.threshold + 0.1
        else:
            threshold = args.threshold
        
        threshold = min(args.threshold + 0.1, 0.6)

        mask = (max_sims > threshold)         # (B,), bool
        if mask.sum() == 0:
            print("No positive samples in this batch, skip...")
            continue

        logits_proto = sims[mask] # shape [pos_n, num_prototypes]
        logits_proto /= 0.1  # temperature scaling

        # scaling temperatures for the selected prototypes


        pos_idx = logits_proto.argmax(dim=1)
        pos_sim = logits_proto.gather(1, pos_idx.unsqueeze(1))
        info_loss = -torch.log(torch.exp(pos_sim) / torch.exp(logits_proto).sum(dim=1, keepdim=True))
        
        soft_assign = F.softmax(logits_proto, dim=1)
        proto_usage = soft_assign.mean(dim=0)  # [K]
        balance_loss = ((proto_usage - 1.0 / 10) ** 2).sum()

        loss = info_loss.mean() + 0.1 * decor_loss + 0.1 * balance_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        total_samples += mask.sum().item()

    avg_epoch_train_loss = epoch_train_loss / total_samples
    print(f"Epoch [{epoch}/{args.proto_epoch}]  Loss: {avg_epoch_train_loss}")

    return encoder, projection, avg_epoch_train_loss


def prototype_contrastive_eval(encoder, projection, proto_test_loader, molecule_id, proto_label):
    encoder.eval()
    projection.eval()
    all_embeddings = []
    all_labels = []
    id2idx = {pid.item(): idx for idx, pid in enumerate(molecule_id)}

    with torch.no_grad():
        for data in proto_test_loader:
            # move the data into cuda
            data = data.to(device)

            query = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            query = projection(query)
            query = F.normalize(query, dim=-1)

            proto_centroids = F.normalize(proto_centroids, dim=-1)

            sims = torch.mm(query, proto_centroids.t())  # [batch, num_pos]
            max_sims, assigned_idx = sims.max(dim=1)
            # print(max(max_sims).item())

            mask = (max_sims > 0.6)
            if mask.sum() == 0:
                print("No positive samples in this batch, skip...")
                continue

            select_query = query[mask]
            select_assigned_idx = assigned_idx[mask]

            all_embeddings.append(select_query.cpu())
            all_labels.append(torch.tensor(select_assigned_idx.cpu(), dtype=torch.long))
        
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    if len(np.unique(all_labels)) < 2:
        print("Not enough clusters for silhouette score calculation.")
        return -1

    sc_cosine = silhouette_score(all_embeddings, all_labels, metric='cosine')
    print(f"Silhouette Coefficient - Cosine: {sc_cosine:.4f}")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    emb_2d = reducer.fit_transform(all_embeddings)

    proto_labels = all_labels  # 该标签表示每个样本对应的prototype类别

    umap_df = pd.DataFrame(emb_2d, columns=['UMAP1', 'UMAP2'])
    umap_df['Prototype'] = proto_labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Prototype', palette='tab10', s=50, edgecolor=None, alpha=0.7)

    plt.title('UMAP of Graph Embeddings by Prototype', fontsize=16)
    plt.xlabel('UMAP 1', fontsize=14)
    plt.ylabel('UMAP 2', fontsize=14)

    plt.legend(title='Prototype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./V3/plots/test_results.png', dpi=600)


    return sc_cosine


def main():
    data_path = './data/all_data.pkl'
    file_path = f"./checkpoints/{args.training_types}_model_{args.models}.pth"

    # load data
    positive_samples_126, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples = load_data(data_path)
    all_pos_samples = pos_train_samples + pos_test_samples

    # check and get the representation model checkpoint
    model = get_representation_model(file_path, pos_train_samples, pos_test_samples)

    # get the prototypes table
    if args.task == 'train':
        prototypes_table = get_prototypes(model, all_pos_samples, unlabeled_samples) # get the prototypes during training process

    print(prototypes_table)
    
    # train the prototype contrastive learning model
    proto_train_samples = pos_train_samples + unl_train_samples
    proto_test_samples = pos_test_samples + unl_test_samples
    proto_train_loader = DataLoader(proto_train_samples, batch_size=args.pcl_batch_size, shuffle=True)
    proto_test_loader = DataLoader(proto_test_samples, batch_size=args.pcl_batch_size, shuffle=False)
    pos_loader = DataLoader(all_pos_samples, batch_size=args.pcl_batch_size, shuffle=False)

    # training data
    molecule_id = prototypes_table['molecule_id'].values.tolist()
    proto_label = prototypes_table['prototypes'].values.tolist()
    molecule_id = torch.tensor(molecule_id, dtype=torch.int)
    proto_label = torch.tensor(proto_label, dtype=torch.int)

    encoder = GINE(num_node_features=proto_train_samples[0].n_node_features, num_edge_features=proto_train_samples[0].n_edge_features, 
        hidden_channels=args.pcl_hidden_channels,
        num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
    projection = ProjectionHead_PCL(in_dim=args.pcl_hidden_channels).to(device)

    # # load ucl model parameters
    # encoder.load_state_dict(model.state_dict())
    # for param in encoder.parameters():
    #     param.requires_grad = True

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projection.parameters()), lr=args.pcl_learning_rate, weight_decay=5e-4)

    
    proto_train_loss = []
    for epoch in tqdm(range(1, args.proto_epoch + 1), desc='Training the prototype contrastive learning model...'):
        # training
        encoder, projection, avg_epoch_train_loss = prototype_contrastive_training(epoch, encoder, projection, optimizer, proto_train_loader, molecule_id, proto_label, pos_loader)
        proto_train_loss.append(avg_epoch_train_loss)

        # evaluating
        if epoch % 5 == 0:
            sc_cosine = prototype_contrastive_eval(encoder, projection, proto_train_loader, molecule_id, proto_label)
        
    plot_train_loss(args.proto_epoch, proto_train_loss, args.models, args.proto_training_types)

    # torch.save(encoder.state_dict(), f'./checkpoints/{args.proto_models}_epoch_{args.proto_epoch}_r_{args.r}.pth')




if __name__=="__main__":
    main()

        


