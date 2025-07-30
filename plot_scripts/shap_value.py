import shap
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU, MultiheadAttention, LayerNorm
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool
from torch_geometric.data import Data

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--d_ff', type=int, default=512, help='Number of model dimension')
parser.add_argument('--epoch', type=int, default=300, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--folds', type=int, default=10, help='Fold number of cross validation')
parser.add_argument('--repeats', type=int, default=5, help='Repeat number of cross validation')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
parser.add_argument('--training_methods', type=str, default='Self_Training', help='Training methods')
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold of self training')
parser.add_argument('--warm_up_epoch', type=int, default=40, help='Self training warm up epoch period')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension of attention')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
parser.add_argument('--desp_dim', type=int, default=217, help='Number of descriptors')
parser.add_argument('--d_keys', type=int, default=128, help='Number of descriptors')
parser.add_argument('--d_values', type=int, default=128, help='Number of descriptors')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

class DANLayer(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, d_keys, d_values, d_ff, dropout):
        super(DANLayer, self).__init__()

        self.query_projection = Linear(hidden_channels, hidden_channels)
        self.key_projection = Linear(hidden_channels, hidden_channels)
        self.value_projection = Linear(hidden_channels, hidden_channels)

        self.multiheadattn = MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.attn_norm = LayerNorm(hidden_channels)

        self.lin1 = Linear(hidden_channels, d_ff)
        self.relu = ReLU()
        self.lin2 = Linear(d_ff, hidden_channels)
        self.dropout = Dropout(dropout)
        self.ffn_attn_norm = LayerNorm(hidden_channels)
    
    def feedforward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        return x

    def forward(self, x):
        q = self.query_projection(x)
        k = self.query_projection(x)
        v = self.value_projection(x)
        x_in = x

        x, _ = self.multiheadattn(q, k, v)
        x = x_in + self.attn_norm(x) # [B, 1, d_keys * num_heads]

        x_ffn = x
        x = self.feedforward(x)
        x = x_ffn + self.ffn_attn_norm(x)

        return x

class DAN(torch.nn.Module): # Descriptor Attention Network
    def __init__(self, hidden_channels, num_heads, desp_dim, d_ff, dropout, d_keys, d_values):
        super(DAN, self).__init__()

        self.desp_num = desp_dim

        self.multihead_attn = MultiheadAttention(hidden_channels, num_heads, batch_first=True)
        self.desp_embed = Linear(desp_dim, hidden_channels * desp_dim)
        self.desp_num = desp_dim
        self.attn_norm = LayerNorm(hidden_channels)

        self.lin1 = Linear(hidden_channels, d_ff)
        self.relu = ReLU()
        self.lin2 = Linear(d_ff, hidden_channels)
        self.dropout = Dropout(dropout)
        self.ffn_attn_norm = LayerNorm(hidden_channels)

        self.dan_layer1 = DANLayer(hidden_channels, num_heads, d_keys, d_values, d_ff, dropout)
        self.dan_layer2 = DANLayer(hidden_channels, num_heads, d_keys, d_values, d_ff, dropout)
        self.dan_layer3 = DANLayer(hidden_channels, num_heads, d_keys, d_values, d_ff, dropout)
    
    def feedforward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

        return x

    def forward(self, x, descriptors):
        B = x.shape[0]
        H = x.shape[1]
        N = self.desp_num
        
        desp_embed = self.desp_embed(descriptors) 
        desp_embed = desp_embed.view(B, N, H) # [batchsize, num_desp_features * hidden_channels] --> [batchsize, num_desp_features, hidden_channels]
        x_in = x.unsqueeze(1)  # [B, 1, hidden_channels]

        x, _ = self.multihead_attn(x_in, desp_embed, desp_embed)
        x = x_in + self.attn_norm(x) # [B, 1, H]

        x_ffn = x
        x = self.feedforward(x)
        x = x_ffn + self.ffn_attn_norm(x)

        x = self.dan_layer1(x)

        return x

class GINE_descriptor(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes, dropout, args):
        super(GINE_descriptor, self).__init__()

        nn1 = Sequential(Linear(num_node_features, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn1, edge_dim=num_edge_features)

        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv2 = GINEConv(nn2, edge_dim=num_edge_features)

        nn3 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv3 = GINEConv(nn3, edge_dim=num_edge_features)

        self.multihead_attn = MultiheadAttention(args.hidden_channels, args.num_heads, batch_first=True)
        self.desp_embed = Linear(args.desp_dim, args.hidden_channels * args.desp_dim)
        self.desp_num = args.desp_dim
        self.attn_norm = LayerNorm(args.hidden_channels)

        self.ffn_lin1 = Linear(hidden_channels, args.d_ff)
        self.relu = ReLU()
        self.ffn_lin2 = Linear(args.d_ff, hidden_channels)
        self.ffn_dropout = Dropout(dropout)
        self.ffn_attn_norm = LayerNorm(hidden_channels)

        self.dan1 = DAN(hidden_channels, args.num_heads, args.desp_dim, args.d_ff, dropout, args.d_keys, args.d_values)
        
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(dropout)

    def feedforward(self, x):
        x = self.ffn_lin1(x)
        x = self.relu(x)
        x = self.ffn_dropout(x)
        x = self.ffn_lin2(x)

        return x

    def forward(self, x, edge_index, edge_attr, batch, descriptors):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)# [batchsize, hidden_channels]

        x = self.dan1(x, descriptors)
        # desp_out, _ = self.multihead_attn(desp_embed, x_in, x_in) # [B, N, H]
        x = x.squeeze(1)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x
    
def model_predict(descriptor_inputs):
    """
    descriptor_inputs: np.array shape (batch_size, 217)
    返回: np.array shape (batch_size, num_classes) 模型预测结果（未经softmax或经过softmax均可，注意对应shap用法）
    """
    model.eval()
    outputs = []
    with torch.no_grad():
        descriptors_tensor = torch.tensor(descriptor_inputs, dtype=torch.float32).unsqueeze(1).to(device)  # [B,1,217]
        for desc in descriptors_tensor:
            out = model(
                fixed_graph.x.to(device),
                fixed_graph.edge_index.to(device),
                fixed_graph.edge_attr.to(device),
                torch.zeros(fixed_graph.x.size(0), dtype=torch.long).to(device),  # batch 全0，因为一个图
                desc.unsqueeze(0)
            )
            out = out.cpu().numpy()
            outputs.append(out)
    return np.vstack(outputs)

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

model = GINE_descriptor(num_node_features=all_data[0].n_node_features, num_edge_features=all_data[0].n_edge_features, 
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)
model.load_state_dict(torch.load('./checkpoints/best_model.pth')) # load the checkpoints
print('Model is loaded!')
model.eval()

fixed_graph = all_data[0]  
descriptors_variants = desp_data[:100]  
background_inputs = descriptors_variants.cpu().numpy()  
test_inputs = desp_data[100:250].cpu().numpy()  
explainer = shap.KernelExplainer(model_predict, background_inputs)
shap_values = explainer.shap_values(test_inputs)
shap_vals_for_class1 = shap_values[:, :, 1]

feature_name_df = pd.DataFrame(pd.read_excel('./plot_scripts/pearson_data/chemical_attributes.xlsx'))
features_name = feature_name_df['Attributes'].values.tolist()
feature_names = [features_name[i] for i in range(test_inputs.shape[1])]
shap.summary_plot(shap_vals_for_class1, test_inputs, feature_names=feature_names)
plt.savefig('./figs/shap_beeswarm.png', dpi=600) 