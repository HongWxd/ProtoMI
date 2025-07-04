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

# # 示例：一组 SMILES
# smiles_list = ["C1=CC=CC=C1"]
# encoder = SmilesEncoder(smiles_list)
# encoded_smiles = encoder.encode_many(smiles_list)
# decoded_smiles = encoder.decode_many(encoded_smiles)
# print(len(encoded_smiles[0]), decoded_smiles)

# with open('./data/norm_normal.pkl', 'rb') as f:
#     desp_data = pickle.load(f)

# with open('./data/all_data.pkl', 'rb') as f:
#     all_data = pickle.load(f)

# desp_data = torch.tensor(desp_data, dtype=torch.float)

# train_data = []
# for desp, graph in zip(desp_data, all_data):
#     graph.descriptors = desp.unsqueeze(0)
#     train_data.append(graph)

#     break

pred = pd.DataFrame(pd.read_csv('./data/predict_1.csv'))
formula_list = pd.DataFrame(set(pred['formula'].values.tolist()))
formula_list.to_csv('./selected_formula.csv', index=False, header=False)
print(len(formula_list))
