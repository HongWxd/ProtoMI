import os
import torch
import copy
import torch.nn.functional as F

from tqdm import tqdm
from model import ProjectionHead, Cluster_GINE, Cluster_GAT, Cluster_GCN, Cluster_GIN
from torch_geometric.loader import DataLoader
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from utils.tools import info_nce_loss, perturb_edges, try_multiple_cluster_combinations, plot_train_loss



class USL():
    def __init__(self, args):
        self.args = args
        self.usl_batch_size = args.usl_batch_size
        self.usl_learning_rate = args.usl_learning_rate
        self.epoch = args.epoch
        self.save_path = args.save_path
        self.training_types = args.training_types
        self.models = args.models
        self.device = args.device
        self.num_classes = args.num_classes
        self.usl_trials = args.usl_trials
        self.save_path = args.save_path
        self.usl_hidden_channels = args.usl_hidden_channels
        self.dropout = args.dropout
        self.retrain_usl = args.retrain_usl
        self.usl_backbone = args.usl_backbone


    def unsupervised_training(self, pos_train_samples, pos_test_samples):
        # Unsupervised training
        # train a GNN model to represent all positive training data and get the prototypes
        pos_train_samples = pos_train_samples + pos_test_samples
        train_loader = DataLoader(pos_train_samples, batch_size=self.usl_batch_size, shuffle=True)
        
        if self.usl_backbone == 'GINE':
            model = Cluster_GINE(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
                hidden_channels=self.usl_hidden_channels, num_classes=self.num_classes, dropout=self.dropout).to(self.device)
        elif self.usl_backbone == 'GAT':
            model = Cluster_GAT(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
                hidden_channels=self.usl_hidden_channels, num_classes=self.num_classes, dropout=self.dropout).to(self.device)
        elif self.usl_backbone == 'GCN':
            model = Cluster_GCN(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
                hidden_channels=self.usl_hidden_channels, num_classes=self.num_classes, dropout=self.dropout).to(self.device)
        elif self.usl_backbone == 'GIN':
            model = Cluster_GIN(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
                hidden_channels=self.usl_hidden_channels, num_classes=self.num_classes, dropout=self.dropout).to(self.device)
        
        projection_head1 = ProjectionHead(in_dim=self.usl_hidden_channels).to(self.device)
        projection_head2 = ProjectionHead(in_dim=self.usl_hidden_channels).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.usl_learning_rate, weight_decay=5e-4)
        
        unsuper_train_loss = []
        silhouette_scores = 0
        best_model = None
        for epoch in tqdm(range(1, self.epoch + 1), desc='Training the representation GNN...'):
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(self.device)

                # graph augmentation: for constractive learning 
                data_aug1 = data.clone() 
                data_aug2 = perturb_edges(data.clone(), self.device) 
                
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
            eval_loader = DataLoader(pos_train_samples, batch_size=self.usl_batch_size, shuffle=False)
            all_embeddings = []
            pos_additives_names = []
            with torch.no_grad():
                for data in eval_loader:
                    data = data.to(self.device)
                    pos_additives_names += data.id
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    all_embeddings.append(out.cpu())

            all_embeddings = F.normalize(torch.cat(all_embeddings), dim=-1).numpy()

            # hierarchical cluster
            Z = linkage(all_embeddings, method='average', metric='cosine')

            # get the best cluster number of all positive samples
            # best_cluster_num, labels = try_multiple_cluster_combinations(Z, all_embeddings, args)
            best_k, labels, all_embeddings, _, _ = try_multiple_cluster_combinations(Z, all_embeddings, pos_additives_names, self.args)
            sil = silhouette_score(all_embeddings, labels, metric='cosine')


            if sil > silhouette_scores:
                silhouette_scores = sil
                best_model = copy.deepcopy(model)
                print(f'Update! Epoch: {epoch}, silhouette score: {sil}')

            print(f"Epoch [{epoch}/{self.epoch}]  Loss: {avg_loss}")
        plot_train_loss(self.epoch, unsuper_train_loss, self.models, self.training_types)

        return best_model, silhouette_scores

    def get_representation_model(self, pos_train_samples, pos_test_samples):
        file_path = f"{self.save_path}/USL_encoder_{self.epoch}_{self.usl_backbone}.pth"        
        if os.path.exists(file_path):
            print(f"Loading the pretrained model from {file_path}...")

            if self.usl_backbone == 'GINE':
                model = Cluster_GINE(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
                    hidden_channels=self.usl_hidden_channels,
                    num_classes=self.num_classes, dropout=self.dropout).to(self.device)
            elif self.usl_backbone == 'GAT':
                model = Cluster_GAT(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
                    hidden_channels=self.usl_hidden_channels,
                    num_classes=self.num_classes, dropout=self.dropout).to(self.device)
            elif self.usl_backbone == 'GCN':
                model = Cluster_GCN(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
                    hidden_channels=self.usl_hidden_channels,
                    num_classes=self.num_classes, dropout=self.dropout).to(self.device)
            elif self.usl_backbone == 'GIN':
                model = Cluster_GIN(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
                    hidden_channels=self.usl_hidden_channels,
                    num_classes=self.num_classes, dropout=self.dropout).to(self.device)
            
            model.load_state_dict(torch.load(file_path)) # load the checkpoints
            best_model = model
        else:
            print("No pretrained model found. Start unsupervised training...")

            best_model = None
            best_sil_score = -1
            for trial in tqdm(range(self.usl_trials), desc=f'Unsupervised learning trials...'):
                model, silhouette_scores = self.unsupervised_training(pos_train_samples, pos_test_samples)
                if silhouette_scores > best_sil_score:
                    best_sil_score = silhouette_scores
                    best_model = model
                    best_trial = trial + 1
            
            print(f'Best trial from unsupervised learning: {best_trial}')
            print(f'Best silhouette score from unsupervised learning: {best_sil_score}')

            torch.save(best_model.state_dict(), file_path)

        return best_model