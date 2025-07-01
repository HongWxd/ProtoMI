import shap
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from model import GCN, GINE
import matplotlib
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

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

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

class DescriptorModelWrapper:
    def __init__(self, model, fixed_graph_data, device='cpu'):
        self.model = model
        self.graph = fixed_graph_data.to(device)
        self.device = device

    def __call__(self, descriptors_batch_numpy):
        descriptors_batch = torch.tensor(descriptors_batch_numpy, dtype=torch.float32).to(self.device)
        batch_size = descriptors_batch.size(0)

        data_list = []
        for i in range(batch_size):
            new_data = self.graph.clone()
            new_data.descriptors = descriptors_batch[i].unsqueeze(0)
            data_list.append(new_data)

        loader = DataLoader(data_list, batch_size=batch_size)

        preds = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch, data.descriptors)
                pred = torch.softmax(out, dim=1)[:, 1]  # 假设是二分类，取 positive 类
                preds.append(pred.cpu().numpy())
        return np.concatenate(preds)

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

model = GINE(num_node_features=all_data[0].n_node_features, num_edge_features=all_data[0].n_edge_features, 
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes, dropout=args.dropout).to(device)
model.load_state_dict(torch.load('./checkpoints/best_model.pth')) # load the checkpoints
print('Model is loaded!')
model.eval()

graph_sample = all_data[0]  # 用一个代表性的图样本
descriptors_array = np.array([graph_sample.descriptors.squeeze().numpy()])

# 构建背景集（例如从训练集中抽样）
background = np.array([data.descriptors.squeeze().numpy() for data in all_data[:100]])

wrapped_model = DescriptorModelWrapper(model, graph_sample, device)
explainer = shap.Explainer(wrapped_model, background)

shap_values = explainer(descriptors_array)
# shap.plots.waterfall(shap_values[0])  # 单个样本详细解释
# shap.plots.bar(shap_values)


descriptors_batch = np.array([data.descriptors.squeeze().numpy() for data in all_data[:100]])
shap_values_ebm = explainer(descriptors_batch)

shap.plots.beeswarm(shap_values_ebm)
plt.tight_layout()
plt.savefig('./figs/shap_beeswarm.png', dpi=600)