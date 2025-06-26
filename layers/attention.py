import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU, MultiheadAttention, LayerNorm

class DAN(torch.nn.Module): # Descriptor Attention Network
    def __init__(self):
        super(DAN, self).__init__()

    def forward(self, x):
        return x