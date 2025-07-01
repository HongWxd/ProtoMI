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

test_data = [i for i in all_data if i.mask == False]
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

model = GINE_descriptor(num_node_features=test_data[0].n_node_features, num_edge_features=test_data[0].n_edge_features, 
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
model.load_state_dict(torch.load('./checkpoints/best_model.pth')) # load the checkpoints
print('Model is loaded!')

model.eval()
with torch.no_grad():
    all_preds = {}
    for data in tqdm(test_loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.descriptors)
        probs = F.softmax(out, dim=-1)
        confs, preds = probs.max(dim=1)
        high_conf_mask = confs >= args.threshold

        select_preds = preds[high_conf_mask]
        select_data = data[high_conf_mask]
        for data, pred in zip(select_data, select_preds):
            all_preds[data.cid] = pred

searching_space_df = pd.DataFrame(pd.read_csv(args.searching_space_path))
cids_list_1 = []
formulas_1 = []
smiles_1 = []
weight_1 = []
cids_list_0 = []
formulas_0 = []
smiles_0 = []
weight_0 = []
for cid, pred in all_preds.items():
    # for cid, pred in zip(cids, preds):
    cid = cid.item()
    pred = pred.item()
    formula = searching_space_df.loc[searching_space_df['cid'] == cid, 'formula'].values[0]
    smile = searching_space_df.loc[searching_space_df['cid'] == cid, 'SMILES'].values[0]
    weight = searching_space_df.loc[searching_space_df['cid'] == cid, 'weight'].values[0]

    if pred != 1:
        cids_list_0.append(cid)
        formulas_0.append(formula)
        smiles_0.append(smile)
        weight_0.append(weight)
    else:
        cids_list_1.append(cid)
        formulas_1.append(formula)
        smiles_1.append(smile)
        weight_1.append(weight)

pred_1_df = pd.DataFrame()
pred_1_df['cid'] = cids_list_1
pred_1_df['formula'] = formulas_1
pred_1_df['smile'] = smiles_1
pred_1_df['weight'] = weight_1
pred_1_df = pred_1_df[pred_1_df['weight'] <= 500]
pred_1_df = pred_1_df[pred_1_df['weight'] >= 100]
pred_1_df.to_csv('./data/predict_1.csv', index=False)

pred_0_df = pd.DataFrame()
pred_0_df['cid'] = cids_list_0
pred_0_df['formula'] = formulas_0
pred_0_df['smile'] = smiles_0
pred_0_df['weight'] = weight_0
# pred_0_df = pred_0_df[pred_0_df['weight'] <= 200]
# pred_0_df = pred_0_df[pred_0_df['weight'] >= 100]
pred_0_df.to_csv('./data/predict_0.csv', index=False)
