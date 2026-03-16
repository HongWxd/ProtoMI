import os
import torch
import argparse
import pandas as pd
from torch_geometric.loader import DataLoader
from model import GCN, GINE, ProjectionHead, Cluster_GINE, ProjectionHead_PCL
from tqdm import tqdm
import copy
import time
import umap
import pickle
from utils.tools import plot_train_loss, perturb_edges, info_nce_loss, try_multiple_cluster_combinations, plot_PCL_Trials_SC
from utils.graph_augmentation import Graph_Augmentation_Helper
from utils.visualization import show_gnn_fp_consistency_results, plot_hierarchical_cluster_dendrogram, plot_cluster_distribution_UMAP
import numpy as np
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
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
parser.add_argument('--epoch', type=int, default=500, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--training_types', type=str, default='Unsupervised learning', help='training_types')
parser.add_argument('--models', type=str, default='GINE', help='Training models')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension of attention')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
parser.add_argument('--desp_dim', type=int, default=217, help='Number of descriptors')
parser.add_argument('--retrain_usl', type=bool, default=False, help='retrain the usl models')
parser.add_argument('--usl_trials', type=int, default=10, help='Number of trials for unsupervised learning')
parser.add_argument('--save_path', type=str, default='checkpoints', help='')


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
parser.add_argument('--proto_epoch', type=int, default=300, help='Number of training epochs')
parser.add_argument('--r', type=int, default=10000, help='number of randomly select neg prototypes')
parser.add_argument('--proto_training_types', type=str, default='Prototype contrastive learning', help='training_types')
parser.add_argument('--proto_models', type=str, default='GINE', help='model name for PCL')
parser.add_argument('--pcl_hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--pcl_learning_rate', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--pcl_batch_size', type=int, default=1024, help='Batch size for training')
parser.add_argument('--threshold', type=float, default=0.3, help='threshold')
parser.add_argument('--topk', type=int, default=35, help='top k samples for each prototype')
parser.add_argument('--pcl_trials', type=int, default=10, help='Number of trials for unsupervised learning')


# main configs
parser.add_argument('--task', type=str, default='train', help='task types')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


def unsupervised_training(pos_train_samples, pos_test_samples):
    # Unsupervised training
    # train a GNN model to represent all positive training data and get the prototypes
    pos_train_samples = pos_train_samples + pos_test_samples
    train_loader = DataLoader(pos_train_samples, batch_size=args.usl_batch_size, shuffle=True)
    model = Cluster_GINE(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
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
        pos_additives_names = []
        with torch.no_grad():
            for data in eval_loader:
                data = data.to(device)
                pos_additives_names += data.id
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                all_embeddings.append(out.cpu())

        all_embeddings = F.normalize(torch.cat(all_embeddings), dim=-1).numpy()

        # hierarchical cluster
        Z = linkage(all_embeddings, method='average', metric='cosine')

        # get the best cluster number of all positive samples
        # best_cluster_num, labels = try_multiple_cluster_combinations(Z, all_embeddings, args)
        best_k, labels, all_embeddings, _, _ = try_multiple_cluster_combinations(Z, all_embeddings, pos_additives_names, args)
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
        print(f"Loading the pretrained model from {file_path}...")

        model = Cluster_GINE(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
            hidden_channels=args.usl_hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
        model.load_state_dict(torch.load(file_path)) # load the checkpoints
        best_model = model
    else:
        print("No pretrained model found. Start unsupervised training...")

        best_model = None
        best_sil_score = -1
        for trial in tqdm(range(args.usl_trials), desc=f'Unsupervised learning trials...'):
            model, silhouette_scores = unsupervised_training(pos_train_samples, pos_test_samples)
            if silhouette_scores > best_sil_score:
                best_sil_score = silhouette_scores
                best_model = model
                best_trial = trial + 1
        
        print(f'Best trial from unsupervised learning: {best_trial}')
        print(f'Best silhouette score from unsupervised learning: {best_sil_score}')

        torch.save(best_model.state_dict(), f'./{args.save_path}/{args.training_types}_model_{args.models}.pth')

    return best_model


def get_prototypes(model, pos_samples, trial, args):
    # get the original cluster in the positive samples
    projection_head = ProjectionHead(in_dim=args.usl_hidden_channels).to(device)
    pos_sample_loader = DataLoader(pos_samples, batch_size=args.usl_batch_size, shuffle=False)
    model.eval()
    projection_head.eval()
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

    # hierarchical cluster
    Z = linkage(pos_graph_embeddings, method='average', metric='cosine')

    # get the best cluster number of all positive samples
    args.reretrain_usl = False
    best_cluster_num, labels, pos_graph_embeddings, pos_additives_names, Z = try_multiple_cluster_combinations(Z, pos_graph_embeddings, pos_additives_names, args)

    reducer_2d = umap.UMAP(random_state=42)
    umap_embeddings = reducer_2d.fit_transform(pos_graph_embeddings)

    # # plot hierarchical cluster dendrogram
    # plot_hierarchical_cluster_dendrogram(Z, pos_additives_names)

    # plot UMAP cluster distribution
    plot_cluster_distribution_UMAP(best_cluster_num, labels, umap_embeddings, trial, args)

    # # show the consistency analysis results
    # if args.task == 'eval':
    #     show_gnn_fp_consistency_results(pos_additives_names, umap_embeddings)

    additive_id_mapping = pd.read_csv(f'./V3/processed_data/additive_id_mapping.csv')
    
    pos_additives_ids = [i for i in pos_additives_names]
    pos_additives_name = [additive_id_mapping.loc[additive_id_mapping['id'] == i,'name'].values for i in pos_additives_ids]

    prototypes_table = pd.DataFrame()
    prototypes_table['molecule_id'] = pos_additives_ids
    prototypes_table['prototypes'] = labels
    prototypes_table['molecule_name'] = pos_additives_name
    prototypes_table.to_csv(f'./{args.save_path}/proto_table_trial_{trial}.csv', index=False)

    return prototypes_table


def update_proto_centroids(molecule_id, proto_label, encoder, projection, all_pos_samples):
    id2idx = {pid.item(): idx for idx, pid in enumerate(molecule_id)}
    pos_loader = DataLoader(all_pos_samples, batch_size=args.pcl_batch_size, shuffle=False)
    pos_embeddings = []
    with torch.no_grad():
        for data in pos_loader:
            data = data.to(device)
            out = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            emb = projection(out)
            emb = F.normalize(emb, dim=-1)
            pos_embeddings.append(emb.cpu())

    all_pos_samples_embeddings = torch.cat(pos_embeddings, dim=0).numpy()  # shape: [num_molecules, hidden_dim]

    # get the positive samples prototypes
    proto_centroids = [] # [N, D]
    unique_labels = np.unique(proto_label) # 1-N prototypes
    for label in unique_labels:
        select_ids = molecule_id[proto_label == label]
        select_pos_embeddings = [all_pos_samples_embeddings[id2idx[i.item()]] for i in select_ids]
        centroid = np.mean(select_pos_embeddings, axis=0)
        proto_centroids.append(centroid)
    
    proto_centroids = torch.tensor(proto_centroids, dtype=torch.float32)
    proto_centroids = F.normalize(proto_centroids, dim=1)

    return proto_centroids


def prototype_contrastive_training(epoch, encoder, projection, optimizer, proto_train_loader, proto_centroids):    
    encoder.train()
    epoch_train_loss = 0
    total_samples = 0
    num_prototypes = proto_centroids.size(0)
    top_k = args.topk  
    for _, data in enumerate(proto_train_loader):
        # move the data into cuda
        data = data.to(device)
        proto_centroids = proto_centroids.to(device)

        query = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        query = projection(query)
        query = F.normalize(query, dim=-1)
        
        # prototype regularization
        proto_sim = proto_centroids @ proto_centroids.t()
        decor_loss = ((proto_sim - torch.eye(proto_sim.size(0), device=proto_sim.device)) ** 2).mean()

        # compute similarities between query and prototypes
        sims = query @ proto_centroids.t()  # shape [batch_size, num_prototypes]
        
        proto_losses = []
        batch_topk_indices = []
        for i in range(num_prototypes):
            sims_i = sims[:, i]       # [B]
            topk_idx = torch.topk(sims_i, k=top_k).indices
            batch_topk_indices.append(topk_idx)
            logits_i = sims[topk_idx] / 0.1

            pos_sim = logits_i[:, i].unsqueeze(1)
            log_sum = torch.logsumexp(logits_i, dim=1, keepdim=True)
            loss_i = -(pos_sim - log_sum).mean()

            proto_losses.append(loss_i)
        
        proto_loss = torch.stack(proto_losses).mean()
        core_idx = torch.unique(torch.cat(batch_topk_indices)) 

        proto_sim = proto_centroids @ proto_centroids.t()
        decor_loss = ((proto_sim - torch.eye(num_prototypes, device=proto_sim.device)) ** 2).mean()

        loss = proto_loss + decor_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        total_samples += core_idx.numel()

    avg_epoch_train_loss = epoch_train_loss / total_samples
    print(f"Epoch [{epoch}/{args.proto_epoch}]  Loss: {avg_epoch_train_loss}  Samples: {total_samples}")

    return encoder, projection, avg_epoch_train_loss


def prototype_contrastive_eval(encoder, projection, proto_test_loader, proto_centroids):
    encoder.eval()
    projection.eval()
    all_embeddings = []
    all_labels = []
    all_margins = []

    with torch.no_grad():
        for data in proto_test_loader:
            # move the data into cuda
            data = data.to(device)
            proto_centroids = proto_centroids.to(device)

            query = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            query = projection(query)
            query = F.normalize(query, dim=-1)

            sims = query @ proto_centroids.t()     # cosine similarity
            batch_size = sims.size(0)   
            
            batch_size, num_prototypes = sims.size()
            proto_coverage = torch.zeros(num_prototypes, dtype=torch.long)
            top_k = args.topk

            for p in range(num_prototypes):
                sims_p = sims[:, p]                         # [B]
                k = min(top_k, batch_size)
                topk_idx = torch.topk(sims_p, k=k).indices  # [k]

                proto_coverage[p] += k

                core_emb = query[topk_idx]                  # [k, D]
                all_embeddings.append(core_emb.cpu())
                all_labels.append(
                    torch.full((k,), p, dtype=torch.long)
                )

                logits = sims[topk_idx] / 0.1               # [k, K]
                top1 = logits.max(dim=1).values             # [k]
                top2 = logits.topk(2, dim=1).values[:, 1]   # [k]
                margin = top1 - top2                        
                all_margins.append(margin.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_margins = torch.cat(all_margins, dim=0).numpy()

    margin_mean = all_margins.mean()
    margin_std = all_margins.std()
    sc_score = silhouette_score(all_embeddings, all_labels, metric='cosine')

    if len(np.unique(all_labels)) < 2:
        print("Not enough clusters for silhouette score calculation.")
        return -1

    print(f"Margin mean: {margin_mean}, Margin std: {margin_std}")
    print(f"Silhouette Coefficient - Cosine: {sc_score:.6f}")

    return all_embeddings, all_labels, sc_score


def main():
    data_path = './data/all_data.pkl'
    file_path = f"./{args.save_path}/{args.training_types}_model_{args.models}.pth"

    if not os.path.exists(f"./{args.save_path}"):
        os.makedirs(f"./{args.save_path}")

    # load data
    print("Loading data...")
    positive_samples_126, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples = load_data(data_path)
    all_pos_samples = pos_train_samples + pos_test_samples
    print('Loaded data: Positive samples:', len(positive_samples_126), 'Unlabeled samples:', len(unlabeled_samples))

    # check and get the representation model checkpoint
    print("Getting the representation model...")
    model = get_representation_model(file_path, pos_train_samples, pos_test_samples)
    # model = get_representation_model(file_path, pos_train_samples, pos_test_samples)


    total_sc_scores = []
    total_best_encoders = []
    total_best_projections = []
    total_best_embeddings = []
    total_best_labels = []
    total_best_proto_centroids = []
    for trial in tqdm(range(1, args.pcl_trials + 1), desc=f'Prototype contrastive learning trials...'):
        # get the prototypes table
        if args.task == 'train':
            prototypes_table = get_prototypes(model, all_pos_samples, trial, args) # get the prototypes during training process
        
        # train the prototype contrastive learning model
        proto_train_samples = pos_train_samples + unl_train_samples
        proto_test_samples = pos_test_samples + unl_test_samples
        proto_train_loader = DataLoader(proto_train_samples, batch_size=args.pcl_batch_size, shuffle=True)
        proto_test_loader = DataLoader(proto_test_samples, batch_size=args.pcl_batch_size, shuffle=False)

        # training data
        molecule_id = prototypes_table['molecule_id'].values.tolist()
        proto_label = prototypes_table['prototypes'].values.tolist()
        molecule_id = torch.tensor(molecule_id, dtype=torch.int)
        proto_label = torch.tensor(proto_label, dtype=torch.int)


        encoder = GINE(num_node_features=proto_train_samples[0].x.shape[1], num_edge_features=proto_train_samples[0].edge_attr.shape[1], 
            hidden_channels=args.pcl_hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
        projection = ProjectionHead_PCL(in_dim=args.pcl_hidden_channels).to(device)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projection.parameters()), lr=args.pcl_learning_rate, weight_decay=5e-4)

        
        proto_train_loss = []
        best_encoder = None
        best_projection = None
        best_embeddings = None
        best_labels = None
        best_proto_centroids = None
        best_sc_cosine = -1
        for epoch in tqdm(range(1, args.proto_epoch + 1), desc='Training the prototype contrastive learning model...'):
            # get the prototypes embeddings
            new_proto_centroids = update_proto_centroids(molecule_id, proto_label, encoder, projection, all_pos_samples)
            if epoch == 1:
                proto_centroids = new_proto_centroids
            else:
                proto_centroids = 0.999 * proto_centroids + 0.001 * new_proto_centroids
            
            torch.save(proto_centroids, f"/data/hwx/boron/prototype_checkpoints/proto_trial_{trial}_epoch_{epoch}.pth")
            print(f'Prototypes for trial {trial} epoch {epoch} are saved.')

            # training
            encoder, projection, avg_epoch_train_loss = prototype_contrastive_training(epoch, encoder, projection, optimizer, proto_train_loader, proto_centroids)
            proto_train_loss.append(avg_epoch_train_loss)

            # evaluating
            all_embeddings, all_labels, sc_cosine = prototype_contrastive_eval(encoder, projection, proto_test_loader, proto_centroids)

            if sc_cosine > best_sc_cosine:
                best_sc_cosine = sc_cosine
                best_encoder = copy.deepcopy(encoder)
                best_projection = copy.deepcopy(projection)
                best_embeddings = all_embeddings
                best_labels = all_labels
                best_proto_centroids = proto_centroids
                print(f'Update! Epoch: {epoch}, silhouette score: {sc_cosine}')
        
        plot_train_loss(args.proto_epoch, proto_train_loss, args.models, args.proto_training_types)
        print(f'Trial {trial} | Best silhouette score: {best_sc_cosine}')

        total_sc_scores.append(best_sc_cosine)
        total_best_encoders.append(best_encoder)
        total_best_projections.append(best_projection)
        total_best_embeddings.append(best_embeddings)
        total_best_labels.append(best_labels)
        total_best_proto_centroids.append(best_proto_centroids)

    plot_PCL_Trials_SC(total_sc_scores, args.pcl_trials)
    best_trial_idx = np.argmax(total_sc_scores)
    print(f'Best trial: {best_trial_idx + 1} | Best silhouette score: {total_sc_scores[best_trial_idx]}')


    # UMAP visualization of the last epoch
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    emb_2d = reducer.fit_transform(total_best_embeddings[best_trial_idx])

    proto_labels = total_best_labels[best_trial_idx] + 1

    umap_df = pd.DataFrame(emb_2d, columns=['UMAP1', 'UMAP2'])
    umap_df['Prototype'] = proto_labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Prototype', palette='tab10', s=50, edgecolor=None, alpha=0.7)

    plt.title(f'UMAP of Graph Embeddings by Prototype | sc: {total_sc_scores[best_trial_idx]}', fontsize=16)
    plt.xlabel('UMAP 1', fontsize=14)
    plt.ylabel('UMAP 2', fontsize=14)

    plt.legend(title='Prototype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('./V3/plots/test_results.png', dpi=600)

    torch.save(total_best_encoders[best_trial_idx].state_dict(), f'./{args.save_path}/encoder_{args.proto_models}_epoch_{args.proto_epoch}.pth')
    torch.save(total_best_projections[best_trial_idx].state_dict(), f'./{args.save_path}/projection_{args.proto_models}_epoch_{args.proto_epoch}.pth')
    torch.save(total_best_proto_centroids[best_trial_idx], f'./{args.save_path}/proto_centroids_{args.proto_models}_epoch_{args.proto_epoch}.pth')


if __name__=="__main__":
    main()

        


