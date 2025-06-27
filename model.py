import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU, MultiheadAttention, LayerNorm
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool
from layers.attention import DAN, DANLayer

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes, dropout, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch, descriptors):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x

class GINE(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes, dropout, args):
        super(GINE, self).__init__()

        nn1 = Sequential(Linear(num_node_features, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn1, edge_dim=num_edge_features)

        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv2 = GINEConv(nn2, edge_dim=num_edge_features)

        nn3 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv3 = GINEConv(nn3, edge_dim=num_edge_features)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(dropout)

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

        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)

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
        descriptors = torch.nan_to_num(descriptors, nan=0.0, posinf=1e6, neginf=-1e6)# some descriptors are None
        print(descriptors)

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