import torch
import argparse
import pandas as pd
from torch_geometric.loader import DataLoader
from model import GCN, GINE, GINE_descriptor
from tqdm import tqdm
import time
import pickle
from utils.tools import plot_train_results, self_training, training_data_analysis, imbalanced_weights
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=256, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=300, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--folds', type=int, default=10, help='Fold number of cross validation')
parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
parser.add_argument('--training_methods', type=str, default='Self_Training', help='Training methods')
parser.add_argument('--threshold', type=float, default=0.85, help='Threshold of self training')
parser.add_argument('--warm_up_epoch', type=int, default=30, help='Self training warm up epoch period')
parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension of attention')
parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for attention')
parser.add_argument('--desp_dim', type=int, default=217, help='Number of descriptors')

args = parser.parse_args()
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

def train(model, train_data, device, optimizer, epoch, pseudo_thr, args):
    labeled_train_data = [i for i in train_data if i.mask == True]
    unlabeled_train_data = [i for i in train_data if i.mask == False]
    train_loader = DataLoader(labeled_train_data, batch_size=args.batch_size, shuffle=True)

    model.train()
    total_loss = 0
    total_samples = 0
    total_masked, weights = imbalanced_weights(labeled_train_data, train_loader, epoch, device)

    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.descriptors)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        loss = criterion(out, data.y)# labeled loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += int(data.mask.sum())
        iter_loss = loss / int(data.mask.sum())
        print(f'\t  Iteras: {i+1} | Loss: {iter_loss:.7f}')
    
    if args.training_methods == 'Self_Training':
        if epoch >= args.warm_up_epoch:# Warm up for several epoches
            if total_masked <= pseudo_thr*2:# Control the pseudo samples 
                labeled_train_data = self_training(model, labeled_train_data, unlabeled_train_data, device, pseudo_thr, weights, args)
                    
    labeled_train_cid_list = [i.cid for i in labeled_train_data]
    unlabeled_train_data = [i for i in train_data if i.cid not in labeled_train_cid_list]
    train_data = labeled_train_data + unlabeled_train_data

    return total_loss, total_samples, train_loader, train_data

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_samples = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            if data.mask.sum() == 0:
                continue

            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.descriptors)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(out[data.mask], data.y[data.mask])

            pred = out.argmax(dim=1)
            all_preds.append(pred[data.mask].cpu())
            all_labels.append(data.y[data.mask].cpu())
            total_samples += int(data.mask.sum())
            total_loss += loss.item()
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    auc_score = roc_auc_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')

    return auc_score, precision, recall, f1, total_samples, total_loss

with open('./data/all_data_descriptors.pkl', 'rb') as f:
    all_data = pickle.load(f)

best_fold = 0
overall_best_auc = 0
all_metrics = []
all_loss_metrics = []
label_0_list_init = []
label_1_list_init = []
label_mask_list_init = []
label_mask_list_last = []
label_0_list_last = []
label_1_list_last = []
best_model_state_dict = None
kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
    print(f'===== Fold {fold+1} =====')
    train_data = [all_data[i] for i in train_idx]
    test_data = [all_data[i] for i in test_idx]
    training_data_analysis(fold+1, train_data, test_data)# print the label ratio during training
    pseudo_thr = len([i for i in train_data if i.mask == True])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = GINE_descriptor(num_node_features=train_data[0].n_node_features, num_edge_features=train_data[0].n_edge_features, 
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout, args=args).to(device)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    total_loss = []
    total_test_loss = []
    total_test_auc = []
    embedding_snapshots = []
    best_test_auc = 0
    best_test_precision = 0
    best_test_recall = 0
    best_test_f1 = 0
    early_stop_counter = 0 # early stopping varient
    pratical_epoch = 0
    train_start_time = time.time()

    for epoch in tqdm(range(1, args.epoch + 1), desc='Training'):
        total_masked = 0
        label_0 = 0
        label_1 = 0
        for data in train_loader:
            total_masked += int(data.mask.sum())
            label_0 += (data.y == 0).sum().item()
            label_1 += (data.y == 1).sum().item()

        if epoch == 1:
            label_mask_list_init.append(total_masked)
            label_0_list_init.append(label_0)
            label_1_list_init.append(label_1)
        
        start_time = time.time()
        total_train_loss, train_samples, train_loader, update_train_data = train(model, train_data, device, optimizer, epoch, pseudo_thr, args)
        train_data = update_train_data
        train_auc, train_precision, train_recall, train_f1, _, _ = evaluate(model, train_loader, device)
        test_auc, test_precision, test_recall, test_f1, test_samples, test_loss = evaluate(model, test_loader, device)
        end_time = time.time()
        epoch_time = end_time - start_time

        if test_auc >= best_test_auc:
            early_stop_counter = 0
            best_test_auc = test_auc
            best_test_precision = test_precision
            best_test_recall = test_recall
            best_test_f1 = test_f1
            if test_auc > overall_best_auc:
                overall_best_auc = test_auc
                best_model_state_dict = model.state_dict()
                best_fold = fold + 1
        else:
            if epoch > args.warm_up_epoch:
                early_stop_counter += 1
                print(f"Early stop counter: {early_stop_counter} / {args.patience}")
                if early_stop_counter >= args.patience:
                    label_mask_list_last.append(total_masked)
                    label_0_list_last.append(label_0)
                    label_1_list_last.append(label_1)
                    print(f"Early stopping at epoch {epoch - 1} for fold {fold + 1}")
                    pratical_epoch = epoch
                    break  # stop training early

        avg_train_loss = total_train_loss / train_samples
        avg_test_loss = test_loss / test_samples
        total_loss.append(avg_train_loss)
        total_test_loss.append(avg_test_loss)
        total_test_auc.append(test_auc)

        print(f'Fold: {fold+1} | Epoch: {epoch} | Epoch Time: {epoch_time:.4f} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}')
        print(f'Train AUC: {train_auc:.4f} | Train Precision: {train_precision:.4f} | Train Recall: {train_recall:.4f} | Train F1: {train_f1:.4f}')
        print(f'Test AUC: {test_auc:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f}')

    train_end_time = time.time()
    print(f'Best Performance for fold {fold + 1}: Test AUC: {best_test_auc:.4f} | Test Precision: {best_test_precision:.4f} | Test Recall: {best_test_recall:.4f} | Test F1: {best_test_f1:.4f} | Total Train Time: {(train_end_time - train_start_time):.4f} ')

    best_metrics = (best_test_auc, best_test_precision, best_test_recall, best_test_f1)
    all_metrics.append(best_metrics)
    plot_train_results(pratical_epoch, total_loss, total_test_loss, total_test_auc, fold)# plot loss and acc figure

all_metrics = np.array(all_metrics)
mean_metrics = all_metrics.mean(axis=0)
std_metrics = all_metrics.std(axis=0)
mean_label_0_ratio_init = [label_0 / samples for label_0, samples in zip(label_0_list_init, label_mask_list_init)]
mean_label_1_ratio_init = [label_1 / samples for label_1, samples in zip(label_1_list_init, label_mask_list_init)]
mean_label_0_ratio_last = [label_0 / samples for label_0, samples in zip(label_0_list_last, label_mask_list_last)]
mean_label_1_ratio_last = [label_1 / samples for label_1, samples in zip(label_1_list_last, label_mask_list_last)]
print(f"\n===== Cross-validation Result =====")
print(f"Mean AUC: {mean_metrics[0]:.4f} ± {std_metrics[0]:4f}")
print(f"Mean Precision: {mean_metrics[1]:.4f} ± {std_metrics[1]:4f}")
print(f"Mean Recall: {mean_metrics[2]:.4f} ± {std_metrics[2]:4f}")
print(f"Mean F1: {mean_metrics[3]:.4f} ± {std_metrics[3]:4f}")
print(f'Mean Label Ratio Init (0:1): {np.mean(mean_label_0_ratio_init)} : {np.mean(mean_label_1_ratio_init)}')
print(f'Mean Label Ratio Last (0:1): {np.mean(mean_label_0_ratio_last)} : {np.mean(mean_label_1_ratio_last)}')
print(f'Mean Label Samples Init: {np.mean(label_mask_list_init)}')
print(f'Mean Label Samples Last: {np.mean(label_mask_list_last)}')
print(f'Best fold: {best_fold}')

model_df = pd.DataFrame(all_metrics)
model_df.columns = ['AUC', 'Precision', 'Recall', 'F1']
model_df.to_csv(f'./plot_scripts/plot_data/GINE_SSL_descriptor_data.csv', index=False)

if best_model_state_dict is not None:
    torch.save(best_model_state_dict, './checkpoints/best_model.pth')
