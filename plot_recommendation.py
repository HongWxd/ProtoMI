import torch
import argparse
import pickle
from torch_geometric.loader import DataLoader
from model import ProjectionHead_PCL, GINE, Cluster_GINE, ProjectionHead
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from utils.graph_augmentation import Graph_Augmentation_Helper
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="Train a GCN model")
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
parser.add_argument('--proto_epoch', type=int, default=300, help='Number of training epochs')
parser.add_argument('--r', type=int, default=10000, help='number of randomly select neg prototypes')
parser.add_argument('--proto_training_types', type=str, default='Prototype contrastive learning', help='training_types')
parser.add_argument('--proto_models', type=str, default='GINE', help='model name for PCL')
parser.add_argument('--pcl_hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--pcl_learning_rate', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--pcl_batch_size', type=int, default=1024, help='Batch size for training')
parser.add_argument('--threshold', type=float, default=0.3, help='threshold')
parser.add_argument('--topk', type=int, default=25, help='top k samples for each prototype')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


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

data_path = './data/all_data.pkl'
file_path = f"./checkpoints/{args.training_types}_model_{args.models}.pth"

# load data
positive_samples_126, unlabeled_samples, pos_train_samples, pos_test_samples, unl_train_samples, unl_test_samples = load_data(data_path)

pos_loader = DataLoader(pos_train_samples + pos_test_samples, batch_size=args.pcl_batch_size, shuffle=False)
unl_loader = DataLoader(unl_train_samples + unl_test_samples, batch_size=args.pcl_batch_size, shuffle=False)


encoder = GINE(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
        hidden_channels=args.pcl_hidden_channels,
        num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
projection = ProjectionHead_PCL(in_dim=args.pcl_hidden_channels).to(device)
encoder.load_state_dict(torch.load(f'./checkpoints/encoder_{args.proto_models}_epoch_{args.proto_epoch}.pth')) # load the checkpoints
projection.load_state_dict(torch.load(f'./checkpoints/projection_{args.proto_models}_epoch_{args.proto_epoch}.pth')) # load the checkpoints
proto_centroids = torch.load(f'./checkpoints/proto_centroids_{args.proto_models}_epoch_{args.proto_epoch}.pth')

model = Cluster_GINE(num_node_features=pos_train_samples[0].x.shape[1], num_edge_features=pos_train_samples[0].edge_attr.shape[1], 
            hidden_channels=args.usl_hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
model.load_state_dict(torch.load(file_path)) # load the checkpoints
projection_head = ProjectionHead(in_dim=args.usl_hidden_channels).to(device)

print(f'Model is loaded!')


def get_embeddings(encoder, projection, dataloader, proto_centroids):
    encoder.eval()
    projection.eval()
    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            data = data.to(device)
            proto_centroids = proto_centroids.to(device)

            query = encoder(data.x, data.edge_index, data.edge_attr, data.batch)
            query = projection(query)
            query = F.normalize(query, dim=-1)
            all_ids.append(data.id.cpu())
            all_embeddings.append(query.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_ids = torch.cat(all_ids, dim=0)

    return all_embeddings, all_ids


def get_proto_center_embeddings(encoder, projection, proto_centroids):
    encoder.eval()
    projection.eval()
    all_embeddings = []

    with torch.no_grad():
        proto_centroids = proto_centroids.to(device)

        query = encoder(proto_centroids.x, proto_centroids.edge_index, proto_centroids.edge_attr, proto_centroids.batch)
        query = projection(query)
        query = F.normalize(query, dim=-1)
        all_embeddings.append(query.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_ids = torch.cat(all_ids, dim=0)

    return all_embeddings


emb_pos, pos_ids = get_embeddings(model, projection_head, pos_loader, proto_centroids)
emb_unl, unl_ids = get_embeddings(encoder, projection, unl_loader, proto_centroids)


all_emb = torch.cat([proto_centroids, emb_pos, emb_unl], dim=0)
proto_id_list = -torch.arange(1, 8)
all_ids = torch.cat([proto_id_list, pos_ids, unl_ids], dim=0)


reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
emb_2d = reducer.fit_transform(all_emb.numpy())


umap_df = pd.DataFrame(emb_2d, columns=['UMAP1', 'UMAP2'])
umap_df['ID'] = all_ids.numpy()
umap_df.to_csv(f'./result_files/umap_locations_all_proto.csv', index=False)
