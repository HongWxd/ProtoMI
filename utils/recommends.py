import torch
import pandas as pd
import torch.nn.functional as F

from utils.data_loader import load_data
from sklearn.metrics import silhouette_score
from torch_geometric.loader import DataLoader
from model import ProjectionHead_PCL, GINE, Cluster_GINE, ProjectionHead




class Recommender():
    def __init__(self, args, pos_loader, unl_loader):

        self.file_path = f"{args.save_path}/USL_encoder_{args.epoch}.pth"
        self.pcl_batch_size = args.pcl_batch_size
        self.pcl_hidden_channels = args.pcl_hidden_channels
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.proto_models = args.proto_models
        self.proto_epoch = args.proto_epoch
        self.save_path = args.save_path
        self.usl_hidden_channels = args.usl_hidden_channels
        self.topk = args.topk
        self.method = args.method
        self.EMA = args.EMA
        self.use_decor_loss = args.use_decor_loss
        self.use_topk = args.use_topk
        self.pos_loader = pos_loader
        self.unl_loader = unl_loader
        self.single_sample = next(iter(unl_loader)).to_data_list()[0]
        self.device = args.device
        self.recommend_model = args.recommend_model


        self.PCL_encoder, self.PCL_projection, self.USL_encoder, self.USL_projection, self.proto_centroids = self.prepare_model()
    
    
    def prepare_model(self):
        # load PCL model and prototypes
        PCL_encoder = GINE(num_node_features=self.single_sample.x.shape[1], num_edge_features=self.single_sample.edge_attr.shape[1], 
        hidden_channels=self.pcl_hidden_channels,
        num_classes=self.num_classes, dropout=self.dropout).to(self.device)
        PCL_projection = ProjectionHead_PCL(in_dim=self.pcl_hidden_channels).to(self.device)
        PCL_encoder.load_state_dict(torch.load(f'{self.save_path}/PCL_encoder_{self.recommend_model}_ema_{self.EMA}_decor_{self.use_decor_loss}_topk_{self.use_topk}.pth')) # load the checkpoints
        PCL_projection.load_state_dict(torch.load(f'{self.save_path}/PCL_projection_{self.recommend_model}_ema_{self.EMA}_decor_{self.use_decor_loss}_topk_{self.use_topk}.pth')) # load the checkpoints


        proto_centroids = torch.load(f'{self.save_path}/proto_centroids.pth')

        # load USL model
        USL_encoder = Cluster_GINE(num_node_features=self.single_sample.x.shape[1], num_edge_features=self.single_sample.edge_attr.shape[1], 
                    hidden_channels=self.usl_hidden_channels,
                    num_classes=self.num_classes, dropout=self.dropout).to(self.device)
        USL_encoder.load_state_dict(torch.load(f'{self.file_path}')) # load the checkpoints
        USL_projection = ProjectionHead(in_dim=self.usl_hidden_channels).to(self.device)

        print(f'All Models are loaded!')

        return PCL_encoder, PCL_projection, USL_encoder, USL_projection, proto_centroids
    

    def recommend(self, dataloader):
        self.PCL_encoder.eval()
        self.PCL_projection.eval()
        all_embeddings = []
        all_labels = []
        all_ids = []

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                proto_centroids = self.proto_centroids.to(self.device)

                query = self.PCL_encoder(data.x, data.edge_index, data.edge_attr, data.batch)
                query = self.PCL_projection(query)
                query = F.normalize(query, dim=-1)

                if self.method == 'random' or self.method == 'morgan':
                    all_embeddings.append(query.cpu())
                    all_labels.append((-1) * torch.ones(query.size(0), dtype=torch.long))
                    all_ids.append(data.id.cpu())
                elif self.method == 'full_model':
                    sims = query @ proto_centroids.t()     # cosine similarity
                    batch_size = sims.size(0)   
                    
                    batch_size, num_prototypes = sims.size()
                    proto_coverage = torch.zeros(num_prototypes, dtype=torch.long)
                    top_k = self.topk

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
                    
                        all_ids.append(data.id[topk_idx].cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_ids = torch.cat(all_ids, dim=0)

        if self.method == 'full_model':
            # calculate silhouette score
            sc_score = silhouette_score(all_embeddings, all_labels, metric='cosine')
        elif self.method == 'random' or self.method == 'morgan':
            sc_score = None

        return all_embeddings, all_labels, all_ids, sc_score


def do_recommendation(recommender, pos_loader, unl_loader):
    if recommender.method == 'random' or recommender.method == 'morgan':
        emb_unl, unl_labels, unl_ids, unl_sc_score = recommender.recommend(unl_loader)
        emb_pos, pos_labels, pos_ids, pos_sc_score = None, None, None, None
        predict_labels_df = pd.DataFrame({'id': unl_ids.cpu().numpy(), 'label': unl_labels.cpu().numpy()})

        return emb_pos, pos_labels, pos_ids, pos_sc_score, emb_unl, unl_labels, unl_ids, unl_sc_score, predict_labels_df
    elif recommender.method == 'full_model':
        emb_pos, pos_labels, pos_ids, pos_sc_score = recommender.recommend(pos_loader)
        pos_proto_labels = pos_labels + 1
        emb_unl, unl_labels, unl_ids, unl_sc_score = recommender.recommend(unl_loader)
        unl_proto_labels = unl_labels + 1

        print('Positive samples remain:', len(pos_ids))
        print(f'Silhouette Score on Samples: {(pos_sc_score * len(pos_ids) + unl_sc_score * len(unl_ids))  / (len(pos_ids) + len(unl_ids))}')

        print('candidate embeddings shape:', emb_unl.shape)
        predict_labels_df = pd.DataFrame({'id': unl_ids.cpu().numpy(), 'label': unl_proto_labels.cpu().numpy()})

    return emb_pos, pos_labels, pos_ids, pos_sc_score, emb_unl, unl_labels, unl_ids, unl_sc_score, predict_labels_df
