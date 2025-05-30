import torch
import argparse
import pandas as pd
from torch_geometric.loader import DataLoader
from model import GCN, GCN_with_edge_attr
from tqdm import tqdm
import time
import pickle
from utils.tools import plot_loss_acc, unlabeled_weight, self_training
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
parser.add_argument('--training_methods', type=str, default='Self_Training', help='Training methods')
parser.add_argument('--threshold', type=float, default=0.9, help='threshold of self training')
parser.add_argument('--T1', type=int, default=15, help='self training warm up epoch period')
parser.add_argument('--T2', type=int, default=150, help='epoch time period of self training')

args = parser.parse_args()
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, device, optimizer, criterion, epoch, args):
    model.train()
    total_loss = 0
    total_samples = 0
    total_pseudo_loss = 0
    total_pseudo_samples = 0

    for data in train_loader:
        if data.mask.sum() == 0:
            continue

        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out[data.mask], data.y[data.mask])# labeled loss

        if args.training_methods == 'Self_Training':
            loss, pseudo_loss, pseudo_samples = self_training(model, data, loss, out, epoch, criterion, device, args)
        else:
            pseudo_loss = torch.tensor(0.0, device=device, requires_grad=True)
            pseudo_samples = 0
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_pseudo_loss += pseudo_loss.item()
        total_samples += int(data.mask.sum())
        total_pseudo_samples += pseudo_samples
    
    return total_loss, total_samples, total_pseudo_loss, total_pseudo_samples

def evaluate(model, loader, device, criterion):
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
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out[data.mask], data.y[data.mask])

            pred = out.argmax(dim=1)
            all_preds.append(pred[data.mask].cpu())
            all_labels.append(data.y[data.mask].cpu())
            total_samples += int(data.mask.sum())
            total_loss += loss.item()
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')

    return accuracy, precision, recall, f1, total_samples, total_loss

with open('./data/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

best_fold = 0
overall_best_acc = 0
all_metrics = []
all_loss_metrics = []
best_model_state_dict = None
kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
    print(f'\n===== Fold {fold+1} =====')
    train_data = [all_data[i] for i in train_idx]
    test_data = [all_data[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = GCN_with_edge_attr(num_node_features=train_data[0].n_node_features, num_edge_features=train_data[0].n_edge_features, 
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout).to(device)

    # model = GCN_with_edge_attr(num_node_features=train_data[0].n_node_features, num_edge_features=train_data[0].n_edge_features, 
    #     hidden_channels=args.hidden_channels,
    #     num_classes=args.num_classes, dropout=args.dropout).to(device)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = []
    total_test_loss = []
    total_test_acc = []
    embedding_snapshots = []
    best_test_acc = 0
    best_test_precision = 0
    best_test_recall = 0
    best_test_f1 = 0
    early_stop_counter = 0 # early stopping varient
    pratical_epoch = 0
    train_start_time = time.time()

    for epoch in tqdm(range(1, args.epoch + 1), desc='Training'):
        start_time = time.time()
        total_train_loss, train_samples, total_pseudo_loss, total_pseudo_samples = train(model, train_loader, device, optimizer, criterion, epoch, args)
        train_accuracy, train_precision, train_recall, train_f1, _, _ = evaluate(model, train_loader, device, criterion)
        test_accuracy, test_precision, test_recall, test_f1, test_samples, test_loss = evaluate(model, test_loader, device, criterion)
        end_time = time.time()
        epoch_time = end_time - start_time

        if test_accuracy >= best_test_acc:
            early_stop_counter = 0
            best_test_acc = test_accuracy
            best_test_precision = test_precision
            best_test_recall = test_recall
            best_test_f1 = test_f1
            if test_accuracy > overall_best_acc:
                overall_best_acc = test_accuracy
                best_model_state_dict = model.state_dict()
                best_fold = fold + 1
        else:
            if epoch > args.T1:
                early_stop_counter += 1
                print(f"Early stop counter: {early_stop_counter} / {args.patience}")
                if early_stop_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch - 1} for fold {fold + 1}")
                    pratical_epoch = epoch - 1
                    break  # stop training early
        
        avg_train_loss = total_train_loss / train_samples
        if total_pseudo_samples == 0:
            avg_pseudo_loss = 0
        else:
            avg_pseudo_loss = total_pseudo_loss / total_pseudo_samples
        avg_test_loss = test_loss / test_samples
        total_loss.append(avg_train_loss)
        total_test_loss.append(avg_test_loss)
        total_test_acc.append(test_accuracy)

        print(f'Fold: {fold+1} | Epoch: {epoch} | Epoch Time: {epoch_time:.4f} | Train Loss: {avg_train_loss:.4f} | Pseudo Loss: {avg_pseudo_loss:.4f} | Test Loss: {avg_test_loss:.4f}')
        print(f'Train Acc: {train_accuracy:.4f} | Train Precision: {train_precision:.4f} | Train Recall: {train_recall:.4f} | Train F1: {train_f1:.4f}')
        print(f'Test Acc: {test_accuracy:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f}')

    train_end_time = time.time()
    print(f'Best Performance for fold {fold + 1}: Test Acc: {best_test_acc:.4f} | Test Precision: {best_test_precision:.4f} | Test Recall: {best_test_recall:.4f} | Test F1: {best_test_f1:.4f} | Total Train Time: {(train_end_time - train_start_time):.4f} ')

    best_metrics = (best_test_acc, best_test_precision, best_test_recall, best_test_f1)
    all_metrics.append(best_metrics)
    plot_loss_acc(pratical_epoch, total_loss, total_test_loss, total_test_acc, fold)# plot loss and acc figure

all_metrics = np.array(all_metrics)
mean_metrics = all_metrics.mean(axis=0)
std_metrics = all_metrics.std(axis=0)
print(f"\n===== Cross-validation Result =====")
print(f"Mean Accuracy: {mean_metrics[0]:.4f} ± {std_metrics[0]:4f}")
print(f"Mean Precision: {mean_metrics[1]:.4f} ± {std_metrics[1]:4f}")
print(f"Mean Recall: {mean_metrics[2]:.4f} ± {std_metrics[2]:4f}")
print(f"Mean F1: {mean_metrics[3]:.4f} ± {std_metrics[3]:4f}")
print(f'Best fold: {best_fold}')

if best_model_state_dict is not None:
    torch.save(best_model_state_dict, './checkpoints/best_model.pth')
