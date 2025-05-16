import torch
import argparse
import pandas as pd
from torch_geometric.loader import DataLoader
from model import GCN
from tqdm import tqdm
import time
import pickle
from utils.tools import plot_loss_acc, train, evaluate
from sklearn.model_selection import KFold
import numpy as np

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=32, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Value of dropout')
parser.add_argument('--folds', type=int, default=5, help='fold number of cross validation')

args = parser.parse_args()

# with open('./data/train_data.pkl', 'rb') as f:
#     train_data = pickle.load(f)
# with open('./data/val_data.pkl', 'rb') as f:
#     val_data = pickle.load(f)
# with open('./data/test_data.pkl', 'rb') as f:
#     test_data = pickle.load(f)

# train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
# test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

with open('./data/all_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

all_metrics = []
all_loss_metrics = []
kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
    print(f'\n===== Fold {fold+1} =====')
    train_data = [all_data[i] for i in train_idx]
    test_data = [all_data[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = GCN(num_node_features=train_data[0].n_node_features,
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes, dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = []
    total_test_loss = []
    total_test_acc = []
    best_test_acc = 0
    best_test_precision = 0
    best_test_recall = 0
    best_test_f1 = 0
    best_model_state_dict = None 
    train_start_time = time.time()

    for epoch in tqdm(range(1, args.epoch + 1), desc='Training'):
        start_time = time.time()
        total_train_loss, train_samples = train(model, train_loader, device, optimizer, criterion)
        train_accuracy, train_precision, train_recall, train_f1, _, _ = evaluate(model, train_loader, device, criterion)
        test_accuracy, test_precision, test_recall, test_f1, test_samples, test_loss = evaluate(model, test_loader, device, criterion)
        end_time = time.time()
        epoch_time = end_time - start_time

        total_loss.append(total_train_loss)
        total_test_loss.append(test_loss)
        total_test_acc.append(test_accuracy)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_test_precision = test_precision
            best_test_recall = test_recall
            best_test_f1 = test_f1
            best_model_state_dict = model.state_dict()

        print(f'Epoch: {epoch} | Epoch Time: {epoch_time:.4f} | Train Loss: {total_train_loss / train_samples:.4f} | Test Loss: {test_loss / test_samples:.4f}')
        print(f'Train Acc: {train_accuracy:.4f} | Train Precision: {train_precision:.4f} | Train Recall: {train_recall:.4f} | Train F1: {train_f1:.4f} | Train Labeled Samples: {train_samples}')
        print(f'Test Acc: {test_accuracy:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f} | Test Labeled Samples: {test_samples}')

    train_end_time = time.time()
    print(f'Best Performance: Test Acc: {best_test_acc:.4f} | Test Precision: {best_test_precision:.4f} | Test Recall: {best_test_recall:.4f} | Test F1: {best_test_f1:.4f} | Total Train Time: {(train_end_time - train_start_time):.4f} ')
    # if best_model_state_dict is not None:
    #     torch.save(best_model_state_dict, './checkpoints/best_model.pth')
    best_metrics = (best_test_acc, best_test_precision, best_test_recall, best_test_f1)
    all_metrics.append(best_metrics)
    plot_loss_acc(args.epoch, total_loss, train_samples, total_test_loss, test_samples, total_test_acc, fold)# plot loss and acc figure

all_metrics = np.array(all_metrics)
mean_metrics = all_metrics.mean(axis=0)
print(f"\n===== Cross-validation Result =====")
print(f"Mean Accuracy: {mean_metrics[0]:.4f}")
print(f"Mean Precision: {mean_metrics[1]:.4f}")
print(f"Mean Recall: {mean_metrics[2]:.4f}")
print(f"Mean F1: {mean_metrics[3]:.4f}")







