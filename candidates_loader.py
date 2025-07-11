import pandas as pd
import numpy as np
import torch
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import argparse
import pickle
from torch_geometric.loader import DataLoader
from model import GCN, GINE, GINE_descriptor
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=300, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--folds', type=int, default=10, help='fold number of cross validation')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--training_methods', type=str, default='Dummy', help='Training methods')
parser.add_argument('--threshold', type=float, default=1.0, help='threshold of self training')
parser.add_argument('--searching_space_path', type=str, default='./data/searching_space_data.csv', help='the path of searching space file')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension of attention')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
parser.add_argument('--desp_dim', type=int, default=217, help='Number of descriptors')
parser.add_argument('--d_keys', type=int, default=128, help='Number of descriptors')
parser.add_argument('--d_values', type=int, default=128, help='Number of descriptors')
parser.add_argument('--d_ff', type=int, default=512, help='Number of model dimension')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

# load the graph data and descriptors data
with open('./data/norm_normal.pkl', 'rb') as f:
    desp_data = pickle.load(f)
with open('./data/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

merged_data = []
desp_data = torch.tensor(desp_data, dtype=torch.float)
for desp, graph in zip(tqdm(desp_data, desc='Loading training data...'), all_data):
    graph.descriptors = desp.unsqueeze(0)
    merged_data.append(graph)
all_data = merged_data

candidates_df = pd.DataFrame(pd.read_csv('./data/candidates/Recommended.csv'))
candidates_cids = candidates_df['CID'].values.tolist()
CASs = candidates_df['CAS'].values.tolist()
Chemical = candidates_df['Chemical'].values.tolist()

candidates_data = [i for i in tqdm(all_data) if i.cid in candidates_cids]
candidates_loader = DataLoader(candidates_data, batch_size=args.batch_size, shuffle=False)

model = GINE_descriptor(num_node_features=candidates_data[0].n_node_features, num_edge_features=candidates_data[0].n_edge_features, 
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
model.load_state_dict(torch.load('./checkpoints/best_model_with_D.pth')) # load the checkpoints
print('Model is loaded!')

model.eval()
with torch.no_grad():
    all_preds = []
    all_confs = []
    all_cids = []
    for data in tqdm(candidates_loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.descriptors)
        probs = F.softmax(out, dim=-1)
        confs, preds = probs.max(dim=1)

        all_candidates = data.cid.cpu().numpy().tolist()
        all_confs = confs.cpu().numpy().tolist()
        all_preds = preds.cpu().numpy().tolist()

feature_metric = []
for cid, pred, confs in zip(all_candidates, all_preds, all_confs):
    features = [i.descriptors for i in candidates_data if i.cid == cid]
    feature_metric.append(features[0].tolist()[0])

feature_name_df = pd.DataFrame(pd.read_excel('./plot_scripts/pearson_data/chemical_attributes.xlsx'))
features_name = feature_name_df['Attributes'].values.tolist()
features_df = pd.DataFrame(feature_metric, columns=features_name)

df = pd.DataFrame()
df['cid'] = all_candidates
df['CAS'] = CASs
df['Chemical'] = Chemical
df['Predicted'] = all_preds
df['Confidence'] = all_confs

results_df = pd.concat([df, features_df], axis=1)
results_df.to_csv('./data/candidates/results.csv', index=False)
# print(results_df)


