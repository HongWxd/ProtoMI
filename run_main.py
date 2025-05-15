import torch
import argparse
import pandas as pd
from torch_geometric.loader import DataLoader
from model import GCN
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from utils.tools import plot_loss_acc

parser = argparse.ArgumentParser(description="Train a GCN model")
parser.add_argument('--analysis', type=bool, default=False, help='Wether to print the summary of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=64, help='Number of hidden channels')
parser.add_argument('--epoch', type=int, default=300, help='Number of training epochs')

args = parser.parse_args()

with open('./data/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('./data/val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)
with open('./data/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = GCN(num_node_features=train_data[0].n_node_features,
            hidden_channels=args.hidden_channels,
            num_classes=args.num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    total_samples = 0
    for data in train_loader:
        if data.mask.sum() == 0:
            continue

        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)

        loss = criterion(out[data.mask], data.y[data.mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += int(data.mask.sum())
    
    return total_loss, total_samples

def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_samples = 0
    with torch.no_grad():
        for data in loader:
            if data.mask.sum() == 0:
                continue

            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            all_preds.append(pred[data.mask].cpu())
            all_labels.append(data.y[data.mask].cpu())
            total_samples += int(data.mask.sum())
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')

    return accuracy, precision, recall, f1, total_samples

total_loss = []
total_test_acc = []
best_test_acc = 0
best_test_precision = 0
best_test_recall = 0
best_test_f1 = 0
best_model_state_dict = None 
train_start_time = time.time()

for epoch in tqdm(range(1, args.epoch + 1), desc='Training'):
    start_time = time.time()
    total_train_loss, train_samples = train()
    train_accuracy, train_precision, train_recall, train_f1, _ = evaluate(train_loader)
    val_accuracy, val_precision, val_recall, val_f1, val_samples = evaluate(val_loader)
    test_accuracy, test_precision, test_recall, test_f1, test_samples = evaluate(test_loader)
    end_time = time.time()
    epoch_time = end_time - start_time

    total_loss.append(total_train_loss)
    total_test_acc.append(test_accuracy)

    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        best_test_precision = test_precision
        best_test_recall = test_recall
        best_test_f1 = test_f1
        best_model_state_dict = model.state_dict()

    print(f'Epoch: {epoch} | Epoch Time: {epoch_time:.4f} | Train Loss: {total_train_loss / train_samples:.4f}')
    print(f'Train Acc: {train_accuracy:.4f} | Train Precision: {train_precision:.4f} | Train Recall: {train_recall:.4f} | Train F1: {train_f1:.4f} | Train Labeled Samples: {train_samples}')
    print(f'Val Acc: {val_accuracy:.4f} | Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f} | Val Labeled Samples: {val_samples}')
    print(f'Test Acc: {test_accuracy:.4f} | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f} | Test Labeled Samples: {test_samples}')

train_end_time = time.time()
print(f'Best Performance: Test Acc: {best_test_acc:.4f} | Test Precision: {best_test_precision:.4f} | Test Recall: {best_test_recall:.4f} | Test F1: {best_test_f1:.4f} | Total Train Time: {(train_end_time - train_start_time):.4f} ')
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, './checkpoints/best_model.pth')

plot_loss_acc(args.epoch, total_loss, total_test_acc)# plot loss and acc figure
