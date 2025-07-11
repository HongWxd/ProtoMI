import numpy as np
import torch
from rdkit import Chem
from smiles_encoder import SmilesEncoder
import matminer.featurizers.composition as mm_composition
import matminer.featurizers.structure as mm_structure
from pymatgen.core import Composition
from ase import Atoms
from dscribe.descriptors import SOAP
from rdkit.Chem import AllChem
import pickle
import pandas as pd
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


smiles_list = ['B(CCC1=CC=CC=C1)(O)O', 'B(C1=CC=CC=C1C#N)(O)O', 'B(C1=CC=C(S1)B(O)O)(O)O', 'B(C1=CC=C(S1)C=O)(O)O', 'B(C1CCCC1)(O)O']
cids = [65389, 2734610, 2770906, 2773430, 2734327]

for cid, smile, i in zip(tqdm(cids), smiles_list, range(1, len(cids) + 1)):
    try:
        mol = read_smiles(smile)

        plt.clf()
        elements = nx.get_node_attributes(mol, name = "element")
        nx.draw(mol, with_labels=True, labels = elements, pos=nx.spring_layout(mol))
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'./figs/{cid}_cluster.png', dpi=300)
    except:
        continue