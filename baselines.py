import pickle
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from tqdm import tqdm
from rdkit import Chem
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

parser = argparse.ArgumentParser(description="Train the machine learning models")
parser.add_argument('--folds', type=int, default=10, help='fold number of cross validation')
parser.add_argument('--model', type=str, default='SVM', help='name of machine learning method')
args = parser.parse_args()

with open('./data/baseline_data.pkl', 'rb') as f:
    all_data = pickle.load(f)

smiles = []
label = []
for data in all_data:
    smiles.append(data.smile)
    label.append(data.y)

# create a list of mols
none_smiles = []
mols = []
y = []
for (i, smile), label in zip(enumerate(smiles), label):
    if Chem.MolFromSmiles(smile) is None:
        none_smiles.append(smile)
    else:
        mols.append(Chem.MolFromSmiles(smile))
        y.append(label)

# create a list of fingerprints from mols
X = [Chem.RDKFingerprint(mol) for mol in tqdm(mols)]

best_fold = 0
all_metrics = []
kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f'\n===== Fold {fold+1} =====')
    X_train, y_train = [X[i] for i in train_idx], [y[i] for i in train_idx]
    X_test, y_test = [X[i] for i in test_idx], [y[i] for i in test_idx]
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    args.model = 'XGB'
    if args.model == 'SVM':
        model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    elif args.model == 'RF':
        model = RandomForestClassifier(max_depth=2, random_state=0)
    elif args.model == 'DT':
        model = tree.DecisionTreeClassifier()
    elif args.model == 'GP':
        kernal = kernel = 1.0 * RBF(1.0)
        model = GaussianProcessClassifier(kernel=kernel, random_state=0)
    elif args.model == 'XGB':
        model = xgb.XGBClassifier(
            objective="binary:logistic", 
            eval_metric="logloss", 
            n_estimators=100,
            max_depth=6,
            learning_rate=0.001,
            random_state=42
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    print(f'Fold: {fold+1} | Test Acc: {acc:.4f} | Test Precision: {precision:.4f} | Test Recall: {recall:.4f} | Test F1: {f1:.4f}')

    metrics = (acc, precision, recall, f1)
    all_metrics.append(metrics)

all_metrics = np.array(all_metrics)
mean_metrics = all_metrics.mean(axis=0)
std_metrics = all_metrics.std(axis=0)
print(f"\n===== Cross-validation Result =====")
print(f'Model: {args.model}')
print(f"Mean Accuracy: {mean_metrics[0]:.4f} ± {std_metrics[0]:4f}")
print(f"Mean Precision: {mean_metrics[1]:.4f} ± {std_metrics[1]:4f}")
print(f"Mean Recall: {mean_metrics[2]:.4f} ± {std_metrics[2]:4f}")
print(f"Mean F1: {mean_metrics[3]:.4f} ± {std_metrics[3]:4f}")
print(f'Best fold: {best_fold+1}')