import os
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

warnings.filterwarnings('ignore')

df = pd.DataFrame(pd.read_csv('./data/predict_1.csv'))
smiles_list = set(df['smile'].values.tolist())
cids = set(df['cid'].values.tolist())

for cid, smile, i in zip(tqdm(cids), smiles_list, range(1, len(cids) + 1)):
    if i == 200:
        break

    try:
        mol = read_smiles(smile)
    except:
        continue

    plt.clf()
    elements = nx.get_node_attributes(mol, name = "element")
    nx.draw(mol, with_labels=True, labels = elements, pos=nx.spring_layout(mol))
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'./figs/ball_stick_1/{cid}.png', dpi=300)


df = pd.DataFrame(pd.read_csv('./data/predict_0.csv'))
smiles_list = set(df['smile'].values.tolist())
cids = set(df['cid'].values.tolist())

for cid, smile, i in zip(tqdm(cids), smiles_list, range(1, len(cids) + 1)):
    if i == 200:
        break

    try:
        mol = read_smiles(smile)
    except:
        continue

    plt.clf()
    elements = nx.get_node_attributes(mol, name = "element")
    nx.draw(mol, with_labels=True, labels = elements, pos=nx.spring_layout(mol))
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'./figs/ball_stick_0/{cid}.png', dpi=300)

