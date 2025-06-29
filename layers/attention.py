import torch
import torch.nn.functional as F
from torch.nn import Linear, MultiheadAttention, LayerNorm, ReLU, Dropout

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
        x = self.dan_layer2(x)
        x = self.dan_layer3(x)

        return x
    






