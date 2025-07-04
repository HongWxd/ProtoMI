import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
from pysmiles import read_smiles
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

def plot_ball_stick():
    # plot possible SEI related molecule
    df = pd.DataFrame(pd.read_csv('./data/predict_1.csv'))
    smiles_list = df['smile'].values.tolist()
    cids = df['cid'].values.tolist()

    for cid, smile, i in zip(tqdm(cids), smiles_list, range(1, len(cids) + 1)):
        try:
            mol = read_smiles(smile)

            plt.clf()
            elements = nx.get_node_attributes(mol, name = "element")
            nx.draw(mol, with_labels=True, labels = elements, pos=nx.spring_layout(mol))
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.savefig(f'./figs/ball_stick_1/{cid}.png', dpi=300)
        except:
            continue

    # # plot possible SEI unrelated molecule
    # df = pd.DataFrame(pd.read_csv('./data/predict_0.csv'))
    # smiles_list = set(df['smile'].values.tolist())
    # cids = set(df['cid'].values.tolist())

    # for cid, smile, i in zip(tqdm(cids), smiles_list, range(1, len(cids) + 1)):
    #     if i == 200:
    #         break

    #     try:
    #         mol = read_smiles(smile)
    #     except:
    #         continue

    #     plt.clf()
    #     elements = nx.get_node_attributes(mol, name = "element")
    #     nx.draw(mol, with_labels=True, labels = elements, pos=nx.spring_layout(mol))
    #     plt.gca().set_aspect('equal')
    #     plt.tight_layout()
    #     plt.savefig(f'./figs/ball_stick_0/{cid}.png', dpi=300)

def plot_distribution():
    df_1 = pd.DataFrame(pd.read_csv('./data/predict_1.csv'))
    smiles_1 = (df_1['smile'].values.tolist())
    # df_0 = pd.DataFrame(pd.read_csv('./data/predict_0.csv'))
    # smiles_0 = (df_0['smile'].values.tolist())
    smiles = smiles_1
    labeled_df = pd.DataFrame(pd.read_csv('./data/labeled_data.csv'))
    labeled_smile = (labeled_df['smile'].values.tolist())
    label = (labeled_df['label'].values.tolist())

    # create a list of mols
    none_smiles = []
    new_smiles = []
    for i, smile in enumerate(smiles):
        if smile.endswith('(O)O'):
            new_smiles.append(smile)
        
        if i > 1500:
            break
    new_smiles = new_smiles + labeled_smile

    mols = []
    for i, smile in enumerate(new_smiles):
        if Chem.MolFromSmiles(smile) is None:
            none_smiles.append(smile)
        else:
            mols.append(Chem.MolFromSmiles(smile))

    smiles = [i for i in smiles if i not in none_smiles]

    # create a list of fingerprints from mols
    fps = [Chem.RDKFingerprint(mol) for mol in tqdm(mols)]

    # normalization
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(fps)

    # t-NSE none-linear
    pca = PCA(n_components=5)
    X_embedded = pca.fit_transform(X_scaled)
    # X_embedded = TSNE(n_components=2, learning_rate='auto',
    #                 init='random', perplexity=3).fit_transform(X_scaled)

    plt.figure()
    colors = ["navy", "turquoise", "darkorange", "yellowgreen"]
    # colors = ["yellowgreen"]
    lw = 2
    # target_names = ['predict_0', 'predict_1', 'SEI_0', 'SEI_1']
    target_names = ['predict_1', 'SEI_1']
    markers = ['o', 'x', 's', 'd']
    # target_names = ['SEI']

    y = []
    for smile in tqdm(new_smiles):
        if smile in smiles_1:
            y.append(1)

    
    for smile, label in zip(tqdm(labeled_smile), label):
        if int(label) == 1:
            y.append(21)
        elif int(label) == 0:
            y.append(20)

    y = np.array(y)
    
    for color, i, target_name, marker in zip(colors, [1, 21], target_names, markers):
        plt.scatter(
            X_embedded[y == i, 0], X_embedded[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name, marker=marker
        )

    for value, smile in zip(X_embedded, new_smiles):
        if smile in labeled_smile:
            continue
        
        print(value, smile)

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("t-SNE of prediction molecule")
    plt.tight_layout()
    plt.savefig('./figs/predict_PCA.jpg', dpi=600)

# plot_ball_stick()
plot_distribution()
