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
    df = pd.DataFrame(pd.read_csv('./data/labeled_UMAP.csv'))
    smiles_list = df['smiles'].values.tolist()
    cids = df['cid'].values.tolist()
    clusters = df['cluster'].values.tolist()

    for cid, smile, cluster, i in zip(tqdm(cids), smiles_list, clusters, range(1, len(cids) + 1)):
        if cluster not in [1, 2, 3, 6, 10]:
            continue

        try:
            mol = read_smiles(smile)

            plt.clf()
            elements = nx.get_node_attributes(mol, name = "element")
            nx.draw(mol, with_labels=True, labels = elements, pos=nx.spring_layout(mol))
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.savefig(f'./figs/real_label/{cid}_cluster_{cluster}.png', dpi=300)
        except:
            continue

plot_ball_stick()

