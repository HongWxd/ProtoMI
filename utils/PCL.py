import os
import torch
import torch.nn.functional as F
import numpy as np
import umap
import copy
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import silhouette_score
from torch_geometric.loader import DataLoader
from scipy.cluster.hierarchy import linkage
from model import GINE, ProjectionHead_PCL, ProjectionHead
from utils.tools import plot_PCL_Trials_SC, plot_train_loss, try_multiple_cluster_combinations
from utils.visualization import plot_cluster_distribution_UMAP


class PCL():
    def __init__(self, args, all_pos_samples):
        self.args = args
        self.device = args.device
        self.pcl_hidden_channels = args.pcl_hidden_channels
        self.usl_hidden_channels = args.usl_hidden_channels
        self.dropout = args.dropout
        self.proto_models = args.proto_models
        self.proto_epoch = args.proto_epoch
        self.save_path = args.save_path
        self.num_classes = args.num_classes
        self.max_cluster = args.max_cluster
        self.proto_training_types = args.proto_training_types
        self.proto_models = args.proto_models
        self.pcl_learning_rate = args.pcl_learning_rate
        self.pcl_trials = args.pcl_trials
        self.topk = args.topk
        self.all_pos_samples = all_pos_samples
        self.usl_batch_size = args.usl_batch_size
        self.pcl_batch_size = args.pcl_batch_size
        self.EMA = args.EMA
        self.use_decor_loss = args.use_decor_loss
        self.use_topk = args.use_topk
        self.recommend_model = args.recommend_model
        self.save_proto_drift = args.save_proto_drift
        self.split_year = args.split_year

        self.is_trained = self.checkpoints_detected()


    def get_prototypes(self, usl_encoder, pos_samples, trial, args):
        # get the original cluster in the positive samples
        usl_projection = ProjectionHead(in_dim=self.usl_hidden_channels).to(self.device)
        pos_sample_loader = DataLoader(pos_samples, batch_size=self.usl_batch_size, shuffle=False)
        usl_encoder = usl_encoder.to(self.device)
        usl_encoder.eval()
        usl_projection.eval()
        pos_graph_embeddings = []
        pos_additives_names = []

        with torch.no_grad():
            for data in pos_sample_loader:
                pos_additives_names += data.id
                data = data.to(self.device)  
                out = usl_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
                emb = usl_projection(out)
                pos_graph_embeddings.append(emb.cpu())
                

        pos_graph_embeddings = F.normalize(torch.cat(pos_graph_embeddings), dim=-1).numpy()  # shape: [num_molecules, hidden_dim]

        # hierarchical cluster
        Z = linkage(pos_graph_embeddings, method='average', metric='cosine')

        # get the best cluster number of all positive samples
        best_cluster_num, labels, pos_graph_embeddings, pos_additives_names, Z = try_multiple_cluster_combinations(Z, pos_graph_embeddings, pos_additives_names, self.args)

        reducer_2d = umap.UMAP(random_state=42)
        umap_embeddings = reducer_2d.fit_transform(pos_graph_embeddings)

        # # plot hierarchical cluster dendrogram
        # plot_hierarchical_cluster_dendrogram(Z, pos_additives_names)

        # # plot UMAP cluster distribution
        # plot_cluster_distribution_UMAP(best_cluster_num, labels, umap_embeddings, trial, self.args)

        additive_id_mapping = pd.read_csv(f'./V3/processed_data/additive_id_mapping.csv')
        
        pos_additives_ids = [i for i in pos_additives_names]
        pos_additives_name = [additive_id_mapping.loc[additive_id_mapping['id'] == i,'name'].values for i in pos_additives_ids]

        prototypes_table = pd.DataFrame()
        prototypes_table['molecule_id'] = pos_additives_ids
        prototypes_table['prototypes'] = labels
        prototypes_table['molecule_name'] = pos_additives_name
        prototypes_table.to_csv(f'./{self.save_path}/proto_table_trial_{trial}.csv', index=False)

        return prototypes_table


    def update_proto_centroids(self, molecule_id, proto_label, encoder, projection):
        id2idx = {pid.item(): idx for idx, pid in enumerate(molecule_id)}
        pos_loader = DataLoader(self.all_pos_samples, batch_size=self.pcl_batch_size, shuffle=False)
        pos_embeddings = []
        with torch.no_grad():
            for data in pos_loader:
                data = data.to(self.device)
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

        print(f'Updated prototype centroids with shape: {proto_centroids.shape}')

        return proto_centroids


    def prototype_contrastive_training(self, epoch, pcl_encoder, pcl_projection, optimizer, proto_train_loader, proto_centroids):    
        pcl_encoder.train()
        pcl_projection.train()
        epoch_train_loss = 0
        total_samples = 0
        num_prototypes = proto_centroids.size(0)
         
        for _, data in enumerate(proto_train_loader):
            # move the data into cuda
            data = data.to(self.device)
            proto_centroids = proto_centroids.to(self.device)

            query = pcl_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            query = pcl_projection(query)
            query = F.normalize(query, dim=-1)
            
            # prototype regularization
            proto_sim = proto_centroids @ proto_centroids.t()
            decor_loss = ((proto_sim - torch.eye(proto_sim.size(0), device=proto_sim.device)) ** 2).mean()

            # compute similarities between query and prototypes
            sims = query @ proto_centroids.t()  # shape [batch_size, num_prototypes]
            topk = self.topk if self.use_topk else sims.size(0)
            
            proto_losses = []
            batch_topk_indices = []
            for i in range(num_prototypes):
                sims_i = sims[:, i]       # [B]
                topk_idx = torch.topk(sims_i, k=topk).indices
                batch_topk_indices.append(topk_idx)
                logits_i = sims[topk_idx] / 0.1

                pos_sim = logits_i[:, i].unsqueeze(1)
                log_sum = torch.logsumexp(logits_i, dim=1, keepdim=True)
                loss_i = -(pos_sim - log_sum).mean()

                proto_losses.append(loss_i)
            
            proto_loss = torch.stack(proto_losses).mean()
            core_idx = torch.unique(torch.cat(batch_topk_indices)) 

            proto_sim = proto_centroids @ proto_centroids.t()

            if self.use_decor_loss:
                decor_loss = ((proto_sim - torch.eye(num_prototypes, device=proto_sim.device)) ** 2).mean()
            else:
                decor_loss = torch.tensor(0.0, device=self.device)

            loss = proto_loss + decor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            total_samples += core_idx.numel()

        avg_epoch_train_loss = epoch_train_loss / total_samples
        print(f"Epoch [{epoch}/{self.proto_epoch}]  Loss: {avg_epoch_train_loss}  Samples: {total_samples}")

        return pcl_encoder, pcl_projection, avg_epoch_train_loss

    
    def prototype_contrastive_eval(self, pcl_encoder, pcl_projection, proto_test_loader, proto_centroids):
        pcl_encoder.eval()
        pcl_projection.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for data in proto_test_loader:
                # move the data into cuda
                data = data.to(self.device)
                proto_centroids = proto_centroids.to(self.device)

                query = pcl_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
                query = pcl_projection(query)
                query = F.normalize(query, dim=-1)

                sims = query @ proto_centroids.t()     # cosine similarity
                batch_size = sims.size(0)   
                
                batch_size, num_prototypes = sims.size()
                proto_coverage = torch.zeros(num_prototypes, dtype=torch.long)

                for p in range(num_prototypes):
                    sims_p = sims[:, p]                         # [B]
                    k = min(self.topk, batch_size)
                    topk_idx = torch.topk(sims_p, k=k).indices  # [k]

                    proto_coverage[p] += k

                    core_emb = query[topk_idx]                  # [k, D]
                    all_embeddings.append(core_emb.cpu())
                    all_labels.append(
                        torch.full((k,), p, dtype=torch.long)
                    )                     


        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        if len(np.unique(all_labels)) < 2:
            print("Not enough clusters for silhouette score calculation.")
            return -1
        
        sc_score = silhouette_score(all_embeddings, all_labels, metric='cosine')
        print(f"Silhouette Coefficient - Cosine: {sc_score:.6f}")

        return all_embeddings, all_labels, sc_score


    def pcl_training(self, usl_encoder, proto_train_samples, proto_test_samples):
        total_sc_scores = []
        total_best_encoders = []
        total_best_projections = []
        total_best_embeddings = []
        total_best_labels = []
        total_best_proto_centroids = []
        for trial in tqdm(range(1, self.pcl_trials + 1), desc=f'Prototype contrastive learning trials...'):
            # get the prototypes table
            prototypes_table = self.get_prototypes(usl_encoder, self.all_pos_samples, trial, self.args) # get the prototypes during training process
            
            # train the prototype contrastive learning model
            proto_train_loader = DataLoader(proto_train_samples, batch_size=self.args.pcl_batch_size, shuffle=True)
            proto_test_loader = DataLoader(proto_test_samples, batch_size=self.args.pcl_batch_size, shuffle=False)

            # training data
            molecule_id = prototypes_table['molecule_id'].values.tolist()
            proto_label = prototypes_table['prototypes'].values.tolist()
            molecule_id = torch.tensor(molecule_id, dtype=torch.int)
            proto_label = torch.tensor(proto_label, dtype=torch.int)

            pcl_encoder = GINE(num_node_features=self.all_pos_samples[0].x.shape[1], num_edge_features=self.all_pos_samples[0].edge_attr.shape[1], 
                hidden_channels=self.pcl_hidden_channels,
                num_classes=self.num_classes, dropout=self.dropout).to(self.device)
            pcl_projection = ProjectionHead_PCL(in_dim=self.pcl_hidden_channels).to(self.device)


            optimizer = torch.optim.Adam(list(pcl_encoder.parameters()) + list(pcl_projection.parameters()), lr=self.pcl_learning_rate, weight_decay=5e-4)

            
            proto_train_loss = []
            best_encoder = None
            best_projection = None
            best_embeddings = None
            best_labels = None
            best_proto_centroids = None
            best_sc_cosine = -1
            for epoch in tqdm(range(1, self.proto_epoch + 1), desc='Training the prototype contrastive learning model...'):
                # get the prototypes embeddings
                new_proto_centroids = self.update_proto_centroids(molecule_id, proto_label, pcl_encoder, pcl_projection)
                if epoch == 1:
                    proto_centroids = new_proto_centroids
                else:
                    if self.EMA:
                        proto_centroids = 0.999 * proto_centroids + 0.001 * new_proto_centroids
                    else:
                        proto_centroids = new_proto_centroids
                

                if self.save_proto_drift:
                    torch.save(proto_centroids, f"/data/hwx/boron/new_proto_save/proto_trial_{trial}_epoch_{epoch}.pth")
                    print(f'Prototypes for trial {trial} epoch {epoch} are saved.')

                # training
                pcl_encoder, pcl_projection, avg_epoch_train_loss = self.prototype_contrastive_training(epoch, pcl_encoder, pcl_projection, optimizer, proto_train_loader, proto_centroids)
                proto_train_loss.append(avg_epoch_train_loss)

                # evaluating
                all_embeddings, all_labels, sc_cosine = self.prototype_contrastive_eval(pcl_encoder, pcl_projection, proto_test_loader, proto_centroids)

                if sc_cosine > best_sc_cosine:
                    best_sc_cosine = sc_cosine
                    best_encoder = copy.deepcopy(pcl_encoder)
                    best_projection = copy.deepcopy(pcl_projection)
                    best_embeddings = all_embeddings
                    best_labels = all_labels
                    best_proto_centroids = proto_centroids
                    print(f'Update! Epoch: {epoch}, silhouette score: {sc_cosine}')
            
            print(f'Trial {trial} | Best silhouette score: {best_sc_cosine}')

            total_sc_scores.append(best_sc_cosine)
            total_best_encoders.append(best_encoder)
            total_best_projections.append(best_projection)
            total_best_embeddings.append(best_embeddings)
            total_best_labels.append(best_labels)
            total_best_proto_centroids.append(best_proto_centroids)

        plot_PCL_Trials_SC(total_sc_scores, self.pcl_trials)
        best_trial_idx = np.argmax(total_sc_scores)
        print(f'Best trial: {best_trial_idx + 1} | Best silhouette score: {total_sc_scores[best_trial_idx]}')

        return total_best_encoders[best_trial_idx], total_best_projections[best_trial_idx], total_best_embeddings[best_trial_idx], total_best_labels[best_trial_idx], total_best_proto_centroids[best_trial_idx]
    
    def load_pcl_encoder_and_projection(self, single_sample):
        PCL_encoder = GINE(num_node_features=single_sample.x.shape[1], num_edge_features=single_sample.edge_attr.shape[1], 
        hidden_channels=self.pcl_hidden_channels,
        num_classes=self.num_classes, dropout=self.dropout).to(self.device)
        PCL_projection = ProjectionHead_PCL(in_dim=self.pcl_hidden_channels).to(self.device)
        PCL_encoder.load_state_dict(torch.load(f'{self.save_path}/PCL_encoder_{self.recommend_model}_ema_{self.EMA}_decor_{self.use_decor_loss}_topk_{self.use_topk}_year_{self.split_year}.pth')) # load the checkpoints
        PCL_projection.load_state_dict(torch.load(f'{self.save_path}/PCL_projection_{self.recommend_model}_ema_{self.EMA}_decor_{self.use_decor_loss}_topk_{self.use_topk}_year_{self.split_year}.pth')) # load the checkpoints

        return PCL_encoder, PCL_projection
    
    def checkpoints_detected(self):
        pcl_encoder_file_path = f'{self.save_path}/PCL_encoder_{self.recommend_model}_ema_{self.EMA}_decor_{self.use_decor_loss}_topk_{self.use_topk}_year_{self.split_year}.pth'
        pcl_projection_file_path = f'{self.save_path}/PCL_projection_{self.recommend_model}_ema_{self.EMA}_decor_{self.use_decor_loss}_topk_{self.use_topk}_year_{self.split_year}.pth'
        return os.path.exists(pcl_encoder_file_path) and os.path.exists(pcl_projection_file_path)