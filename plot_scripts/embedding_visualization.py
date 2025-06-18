import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import argparse
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle
import numpy as np
import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout, Sequential, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GINEConv
from sklearn.model_selection import train_test_split
import imageio.v2 as imageio
import os
from sklearn.utils.class_weight import compute_class_weight

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=300, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--folds', type=int, default=5, help='fold number of cross validation')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--training_methods', type=str, default='Self_Training', help='Training methods')
parser.add_argument('--threshold', type=float, default=0.95, help='threshold of self training')
parser.add_argument('--warm_up_epoch', type=int, default=30, help='self training warm up epoch period')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x

class GCN_with_edge_attr(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_classes, dropout):
        super(GCN_with_edge_attr, self).__init__()

        nn1 = Sequential(Linear(num_node_features, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(nn1, edge_dim=num_edge_features)

        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv2 = GINEConv(nn2, edge_dim=num_edge_features)

        self.lin = Linear(hidden_channels, num_classes)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        return x

def imbalanced_weights(train_data, device):
    y = []
    for data in train_data:
        y.append(data.y.item())
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    return weights

def self_training(model, labeled_train_data, unlabeled_train_data, device, pseudo_thr, weights, args):
    model.eval()
    unlabeled_loader = DataLoader(unlabeled_train_data, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(unlabeled_loader):
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
            probs = F.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=1)
            high_conf_mask = confs > args.threshold

            if high_conf_mask.sum() > 0:
                data.y = data.y.clone()
                data.mask = data.mask.clone()
                data.y[high_conf_mask] = preds[high_conf_mask]
                data.mask[high_conf_mask] = True
                
                update_list = data[high_conf_mask]
                confs_list = confs[high_conf_mask]
                update_list = sample_balancer(update_list, pseudo_thr, confs_list, labeled_train_data)
                for update_data in update_list:
                    if len(labeled_train_data) >= pseudo_thr*2:
                        continue
                    
                    update_data = update_data.cpu()
                    update_data.mask = update_data.mask.item()
                    update_data.cid = update_data.cid.item()
                    update_data.n_nodes = update_data.n_nodes.item()
                    update_data.n_edges = update_data.n_edges.item()
                    update_data.n_node_features = update_data.n_node_features.item()
                    update_data.n_edge_features = update_data.n_edge_features.item()
                    labeled_train_data.append(update_data)
    
    return labeled_train_data

def sample_balancer(update_list, pseudo_thr, confs_list, labeled_train_data):
    pos_sample = 0
    neg_sample = 0
    for data in labeled_train_data:
        if data.y == 0:
            neg_sample += 1
        elif data.y == 1:
            pos_sample += 1

    balanced_update_list = []
    pos_data = []
    neg_data = []
    pos_conf = []
    neg_conf = []
    for data, confs in zip(update_list, confs_list):
        if data.y.item() == 1:
            pos_data.append(data)
            pos_conf.append(confs)
        elif data.y.item() == 0:
            neg_data.append(data)
            neg_conf.append(confs)
    
    pos_need = pseudo_thr - pos_sample
    neg_need = pseudo_thr - neg_sample

    if pos_need >= len(pos_data):
        balanced_update_list += pos_data
    else:
        if pos_need != 0:
            _, pos_topk_indices = torch.topk(torch.tensor(pos_conf), pos_need, largest=True)
            pos_mask = torch.zeros_like(torch.tensor(pos_conf), dtype=torch.bool)
            pos_mask[pos_topk_indices] = True
            select_pos_data = [data for i, data in enumerate(pos_data) if pos_mask[i]]
            balanced_update_list += select_pos_data
    
    if neg_need >= len(neg_data):
        balanced_update_list += neg_data
    else:
        if neg_need != 0:
            print(neg_need)
            _, neg_topk_indices = torch.topk(torch.tensor(neg_conf), neg_need, largest=True)
            neg_mask = torch.zeros_like(torch.tensor(neg_conf), dtype=torch.bool)
            neg_mask[neg_topk_indices] = True
            select_neg_data = [data for i, data in enumerate(neg_data) if neg_mask[i]]
            print(neg_mask)
            print(neg_data)
            balanced_update_list += select_neg_data

    return balanced_update_list

def visualize_embeddings(model, dataloader, epoch, original_train_data, args):
    origin_cids = []
    for origin_data in original_train_data:
        origin_cids.append(origin_data.cid)
    print('origin_cids', len(origin_cids))
    
    model.eval()
    all_embeds = []
    all_labels = []
    all_flags = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            embeds = model(data.x, data.edge_index, data.edge_attr, data.batch)
            all_embeds.append(embeds.cpu())
            all_labels.append(data.y.cpu())

            cids = data.cid.cpu().numpy()
            origin_idx = []
            for i in cids:
                if i in origin_cids:
                    origin_idx.append(True)
                else:
                    origin_idx.append(False)
            all_flags.append(origin_idx)

    embeds = torch.cat(all_embeds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    flags = np.array([item for sublist in all_flags for item in sublist])
    # flags = torch.cat(all_flags, dim=0).numpy()

    reducer = PCA(n_components=2)
    embeds_2d = reducer.fit_transform(embeds)

    plt.figure(figsize=(6, 6))
    num_classes = len(np.unique(labels))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels_map = [
        'Initial class 0',  # origin=True & label=0
        'Initial class 1',  # origin=True & label=1
        'Added class 0',    # origin=False & label=0
        'Added class 1',    # origin=False & label=1
    ]
    for i, (is_origin, class_label) in enumerate([(True, 0), (True, 1), (False, 0), (False, 1)]):
        idx = (flags == is_origin) & (labels == class_label)
        
        plt.scatter(
            embeds_2d[idx, 0], embeds_2d[idx, 1],
            label=labels_map[i],
            alpha=0.7, s=20, color=colors[i]
        )

    # for i in range(num_classes):
    #     idx = labels == i
    #     plt.scatter(embeds_2d[idx, 0], embeds_2d[idx, 1], label=f'Class {i}', alpha=0.7, s=20)

    plt.legend()
    plt.title(f'Graph Embedding at Epoch {epoch}')
    plt.grid(True)
    plt.tight_layout()
    if args.training_methods == 'Self_Training':
        plt.savefig(f'./figs/embedding_evol/embed_epoch_{epoch:03d}_SSL.png')
    else:
        plt.savefig(f'./figs/embedding_evol/embed_epoch_{epoch:03d}.png')
    plt.close()

def SSL_train(model, train_data, device, optimizer, criterion, epoch, pseudo_thr, args):
    labeled_train_data = [i for i in train_data if i.mask == True]
    unlabeled_train_data = [i for i in train_data if i.mask == False]
    train_loader = DataLoader(labeled_train_data, batch_size=args.batch_size, shuffle=True)
    if epoch == 1:
        original_train_data = labeled_train_data
    else:
        original_train_data = None

    model.train()
    total_loss = 0
    total_samples = 0
    total_masked = 0
    label_0 = 0
    label_1 = 0
    for data in train_loader:
        total_masked += int(data.mask.sum())
        label_0 += (data.y == 0).sum().item()
        label_1 += (data.y == 1).sum().item()
    print(f"[Epoch {epoch}] Train set labeled (mask=True): {total_masked} | label 0: {label_0 / total_masked} | label 1: {label_1 / total_masked}")
    weights = imbalanced_weights(labeled_train_data, device)

    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out[data.mask], data.y[data.mask])# labeled loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += int(data.mask.sum())
        iter_loss = loss / int(data.mask.sum())
        print(f'\t  Iteras: {i+1} | Loss: {iter_loss:.7f}')
    
    if epoch >= args.warm_up_epoch:# Warm up for several epoches
        if total_masked <= pseudo_thr*2:# Control the pseudo samples 
            labeled_train_data = self_training(model, labeled_train_data, unlabeled_train_data, device, pseudo_thr, weights, args)
                    
    labeled_train_cid_list = [i.cid for i in labeled_train_data]
    unlabeled_train_data = [i for i in train_data if i.cid not in labeled_train_cid_list]
    train_data = labeled_train_data + unlabeled_train_data
    return train_data, train_loader, original_train_data

def train(model, train_loader, device, optimizer, criterion):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

def make_gif(args, image_folder):
    if args.training_methods == 'Self_Training':
        output_path='./figs/SSL_embedding_evolution.gif'
        postfix = 'SSL.png'
    else:
        output_path='./figs/embedding_evolution.gif'
        postfix = '.png'
    
    images = []
    for epoch in sorted(os.listdir(image_folder)):
        if epoch.endswith(postfix):
            images.append(imageio.imread(os.path.join(image_folder, epoch)))
    imageio.mimsave(output_path, images, duration=0.4)

# main
with open('./data/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42, shuffle=True)
pseudo_thr = len([i for i in train_data if i.mask == True])

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

model = GCN_with_edge_attr(num_node_features=train_data[0].n_node_features, num_edge_features=train_data[0].n_edge_features, 
        hidden_channels=args.hidden_channels,
        num_classes=args.num_classes, dropout=args.dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

original_data = None
for epoch in tqdm(range(1, args.epoch + 1), desc='Training'):
    # train(model, train_loader, device, optimizer, criterion)
    if epoch == 1:
        train_data, train_loader, original_train_data = SSL_train(model, train_data, device, optimizer, criterion, epoch, pseudo_thr, args)
        original_data = original_train_data
    else:
        train_data, train_loader, _ = SSL_train(model, train_data, device, optimizer, criterion, epoch, pseudo_thr, args)

    visualize_embeddings(model, train_loader, epoch, original_data, args)

make_gif(args, image_folder='./figs/embedding_evol')