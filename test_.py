import numpy as np
from rdkit import Chem
from smiles_encoder import SmilesEncoder

# 示例：一组 SMILES
smiles_list = ["C1=CC=CC=C1"]
encoder = SmilesEncoder(smiles_list)
encoded_smiles = encoder.encode_many(smiles_list)
decoded_smiles = encoder.decode_many(encoded_smiles)
print(len(encoded_smiles[0]), decoded_smiles)
