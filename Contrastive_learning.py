import os
import torch
import argparse
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.nn import BCEWithLogitsLoss
from model import GCN, GINE, ProjectionHead, Cluster_GINE
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
parser.add_argument('--proto_epoch', type=int, default=50, help='Number of training epochs')
parser.add_argument('--r', type=int, default=10000, help='number of randomly select neg prototypes')
parser.add_argument('--proto_training_types', type=str, default='Prototype contrastive learning', help='training_types')
parser.add_argument('--proto_models', type=str, default='GINE', help='model name for PCL')
parser.add_argument('--pcl_hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--pcl_learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--pcl_batch_size', type=int, default=1024, help='Batch size for training')


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
        sil = silhouette_score(all_embeddings, labels)

        if sil > silhouette_scores:
            silhouette_scores = sil
            best_model = copy.deepcopy(model)
            print(f'Update! Epoch: {epoch}, silhouette score: {sil}')

        print(f"Epoch [{epoch}/{args.epoch}]  Loss: {avg_loss}")
    plot_train_loss(args.epoch, unsuper_train_loss, args.models, args.training_types)
    torch.save(best_model.state_dict(), f'./checkpoints/{args.training_types}_model_{args.models}.pth')

    return best_model


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
    else:
        model = unsupervised_training(pos_train_samples, pos_test_samples)

    return model


def get_prototypes(model, pos_samples, unlabeled_samples):
    # get the original cluster in the positive samples
    model.eval()
    projection_head = ProjectionHead(in_dim=args.usl_hidden_channels).to(device)
    pos_sample_loader = DataLoader(pos_samples, batch_size=args.usl_batch_size, shuffle=False)
    unl_sample_loader = DataLoader(unlabeled_samples, batch_size=args.usl_batch_size, shuffle=False)
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
            

    pos_graph_embeddings = F.normalize(torch.cat(pos_graph_embeddings), dim=-1).numpy()  # shape: [num_molecules, hidden_dim]
    unl_graph_embeddings = F.normalize(torch.cat(unl_graph_embeddings), dim=-1).numpy()

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

    # get the positive samples prototypes
    proto_centroids = [] # [N, D]
    unique_labels = np.unique(labels) # 1-N prototypes
    for label in unique_labels:
        cluster_points = pos_graph_embeddings[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        proto_centroids.append(centroid)
    
    proto_centroids = np.array(proto_centroids)
    
    dist_matrix = cdist(unl_graph_embeddings, proto_centroids, metric='cosine')  # shape: (180000, 10)
    closest_proto = np.argmin(dist_matrix, axis=1)
    closest_proto += 1

    # get unlabeled samples prototypes
    all_graph_embeddings = np.vstack([pos_graph_embeddings, unl_graph_embeddings]) # [pos + neg, D]
    all_prototypes = np.concatenate([np.array(labels), closest_proto])
    densities = [] # [N, ]
    for label in unique_labels:
        select_embeddings = all_graph_embeddings[all_prototypes == label]
        centroid = proto_centroids[unique_labels == label]

        distances = cdist(select_embeddings, centroid.reshape(1, -1), metric='cosine')
        d = (np.sqrt(distances).mean()) / np.log(len(select_embeddings) + 10)
        densities.append(d)
    
    # clamp extreme values for stability
    low, high = np.percentile(densities, 10), np.percentile(densities, 90)
    densities = np.clip(densities, low, high)

    # === scale the mean to temperature  ===
    densities = args.temperature * densities / densities.mean()
    densities = np.array(densities)

    samples_densities = []
    for label in all_prototypes:
        sample_density = densities[unique_labels == label][0]
        samples_densities.append(sample_density)
    
    pos_additives_ids = [int(i.item()) for i in pos_additives_names]
    unl_additives_ids = [int(i.item()) for i in unl_additives_names]
    molecules_id = pos_additives_ids + unl_additives_ids

    prototypes_table = pd.DataFrame()
    prototypes_table['molecule_id'] = molecules_id
    prototypes_table['prototypes'] = all_prototypes
    prototypes_table['density'] = samples_densities
    prototypes_table.to_csv('./proto_table.csv', index=False)

    # get the prototype embeddings for each sample
    prototypes_emb = proto_centroids[np.array(all_prototypes) - 1]
    unl_proto_embeddings = proto_centroids[closest_proto - 1]

    batch_size = 8192
    all_logits = []
    unl_graph_embeddings = torch.tensor(unl_graph_embeddings).to(device)
    unl_proto_embeddings = torch.tensor(unl_proto_embeddings).to(device)

    for i in range(0, unl_graph_embeddings.size(0), batch_size):
        emb_batch = unl_graph_embeddings[i:i+batch_size]  # [B, D]
        logits_batch = torch.matmul(emb_batch, unl_proto_embeddings.t())  # [B, P]
        all_logits.append(logits_batch.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits / 0.1, dim=1)
    conf, assigned_proto = probs.max(dim=1)
    print(conf.max().item())
    # high_conf_indices = torch.where(conf > 0.7)[0]
    # print(high_conf_indices.sum())


    # umap_model = umap.UMAP(n_components=2, random_state=42)
    # umap_embeddings = umap_model.fit_transform(all_graph_embeddings)  # [N, 2]

    # proto_labels = all_prototypes  # 该标签表示每个样本对应的prototype类别

    # umap_df = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'])
    # umap_df['Prototype'] = proto_labels

    # plt.figure(figsize=(10, 8))
    # sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Prototype', palette='tab10', s=50, edgecolor=None, alpha=0.7)

    # plt.title('UMAP of Graph Embeddings by Prototype', fontsize=16)
    # plt.xlabel('UMAP 1', fontsize=14)
    # plt.ylabel('UMAP 2', fontsize=14)

    # plt.legend(title='Prototype', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig('./V3/plots/labeled_all_samples.png', dpi=600)

    return prototypes_table, prototypes_emb


def prototype_contrastive_training(epoch, model, encoder, projection, optimizer, proto_train_loader, molecule_id, prototype_embeddings, densities, criterion):
    id2idx = {pid.item(): idx for idx, pid in enumerate(molecule_id)}
    
    encoder.train()
    epoch_train_loss = 0
    total_samples = 0
    for i, data in enumerate(proto_train_loader):
        # get the neg prototypes id
        neg_proto_id_all = list(set(id2idx.keys()) - set(data.id.tolist()))

        # move the data into cuda
        data = data.to(device)

        # get all pos and neg prototypes embeddings
        pos_indices = [id2idx[id.item()] for id in data.id]
        pos_prototypes = prototype_embeddings[pos_indices]
        neg_proto_id = torch.tensor(sample(neg_proto_id_all, args.r), dtype=torch.int).to(device)
        neg_indices = [id2idx[id.item()] for id in neg_proto_id]
        neg_prototypes = prototype_embeddings[neg_indices]

        proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0) # [pos_n + neg_n, D]


        query = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        # output = model(data.x, data.edge_index, data.edge_attr, data.batch)

        # emb1 = F.normalize(query, dim=-1)
        # emb2 = F.normalize(output, dim=-1)
        # cosine = F.cosine_similarity(emb1, emb2, dim=1)
        # print("mean cosine encoder1 vs encoder2:", cosine.mean().item())
        # print("std cosine:", cosine.std().item())

        query = projection(query)
        query = F.normalize(query, dim=-1)
        # proto_selected = query.clone()
        proto_selected = F.normalize(proto_selected, dim=-1)

        logits_proto = torch.mm(query, proto_selected.t())
        labels_proto = torch.arange(len(pos_indices), dtype=torch.long).to(device)
        # print('Q:', query)
        
        print('logits', logits_proto)
        logits_subset = logits_proto[:, :len(pos_indices)]
        max_indices = torch.argmax(logits_subset, dim=1)
        print('max indices:', max_indices)
        break


        # # scaling temperatures for the selected prototypes
        # temp_proto_pos = densities[pos_indices]
        # temp_proto_neg = densities[neg_indices]
        # temp_proto_all = torch.cat([temp_proto_pos, temp_proto_neg], dim=0)  # shape [pos_n + neg_n]
        # logits_proto /= temp_proto_all.unsqueeze(0)

        loss = criterion(logits_proto, labels_proto)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        total_samples += len(data)

    avg_epoch_train_loss = epoch_train_loss / total_samples
    print(f"Epoch [{epoch}/{args.proto_epoch}]  Loss: {avg_epoch_train_loss}")

    return encoder, projection, avg_epoch_train_loss


def prototype_contrastive_eval(encoder, projection, proto_test_loader, molecule_id, prototype_embeddings, densities, criterion):
    encoder.eval()
    projection.eval()
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    total_loss = 0
    id2idx = {pid.item(): idx for idx, pid in enumerate(molecule_id)}

    with torch.no_grad():
        for data in proto_test_loader:
            # move the data into cuda
            data = data.to(device)

            query = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            query = projection(query)


            pos_indices = [id2idx[id.item()] for id in data.id]
            pos_prototypes = prototype_embeddings[pos_indices]

            neg_proto_id_all = list(set(id2idx.keys()) - set(data.id.tolist()))
            neg_proto_id = torch.tensor(sample(neg_proto_id_all, args.r), dtype=torch.int).to(device)
            neg_indices = [id2idx[id.item()] for id in neg_proto_id]
            neg_prototypes = prototype_embeddings[neg_indices]


            proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)
            query = F.normalize(query, dim=-1)
            proto_selected = F.normalize(proto_selected, dim=-1)

            # logits
            logits_proto = torch.mm(query, proto_selected.t())

            labels_proto = torch.arange(len(pos_indices), dtype=torch.long).to(device)

            # scaling temperatures
            temp_proto_pos = densities[pos_indices]
            temp_proto_neg = densities[neg_indices]
            temp_proto_all = torch.cat([temp_proto_pos, temp_proto_neg], dim=0)  # shape [pos+neg]
            logits_proto /= temp_proto_all.unsqueeze(0)

            # loss
            loss = criterion(logits_proto, labels_proto)
            total_loss += loss.item()

            # ===== Top-k Evaluation =====
            _, topk_indices = torch.topk(logits_proto, k=5, dim=1)
            top1_correct += (topk_indices[:, 0] == labels_proto).sum().item()
            top5_correct += (topk_indices == labels_proto.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += len(data)


    avg_loss = total_loss / total_samples
    top1_acc = top1_correct / total_samples
    top5_acc = top5_correct / total_samples

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc*100:.2f}%")


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
        prototypes_table, prototype_embeddings = get_prototypes(model, all_pos_samples, unlabeled_samples) # get the prototypes during training process

    # train the prototype contrastive learning model
    proto_train_samples = pos_train_samples + unl_train_samples
    proto_test_samples = pos_test_samples + unl_test_samples
    proto_train_loader = DataLoader(proto_train_samples, batch_size=args.pcl_batch_size, shuffle=True)
    proto_test_loader = DataLoader(proto_test_samples, batch_size=args.pcl_batch_size, shuffle=False)
    
    # training data
    molecule_id = prototypes_table['molecule_id'].values.tolist()
    densities = prototypes_table['density'].values.tolist()
    molecule_id = torch.tensor(molecule_id, dtype=torch.int)
    prototype_embeddings = torch.tensor(prototype_embeddings, dtype=torch.float)
    densities = torch.tensor(densities, dtype=torch.float)

    encoder = GINE(num_node_features=proto_train_samples[0].n_node_features, num_edge_features=proto_train_samples[0].n_edge_features, 
        hidden_channels=args.pcl_hidden_channels,
        num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
    projection = ProjectionHead(in_dim=args.pcl_hidden_channels).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    # load ucl model parameters
    encoder.load_state_dict(model.state_dict())
    for param in encoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projection.parameters()), lr=args.pcl_learning_rate, weight_decay=5e-4)


    prototype_embeddings = prototype_embeddings.to(device)
    densities = densities.to(device)
    
    proto_train_loss = []
    for epoch in tqdm(range(1, args.proto_epoch + 1), desc='Training the prototype contrastive learning model...'):
        # training
        encoder, projection, avg_epoch_train_loss = prototype_contrastive_training(epoch, model, encoder, projection, optimizer, proto_train_loader, molecule_id, prototype_embeddings, densities, criterion)
        proto_train_loss.append(avg_epoch_train_loss)

        # evaluating
        prototype_contrastive_eval(encoder, projection, proto_test_loader, molecule_id, prototype_embeddings, densities, criterion)
        
    plot_train_loss(args.proto_epoch, proto_train_loss, args.models, args.proto_training_types)

    # torch.save(encoder.state_dict(), f'./checkpoints/{args.proto_models}_epoch_{args.proto_epoch}_r_{args.r}.pth')




if __name__=="__main__":
    main()

        


