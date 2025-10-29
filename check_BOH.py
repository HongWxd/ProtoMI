from rdkit import Chem
import pandas as pd
import json

def has_B_OH_bond(smiles):
    if smiles.startswith("B") and smiles.endswith("(O)O"):
        return True
    else:
        return False

searching_space_df = pd.read_csv('./data/searching_space_data_V2.csv')

with open('./V3/processed_data/additives.json', 'r') as f:
    additives_data = json.load(f)

additives_smiles = [i['smiles'] for i in additives_data.values()]
smiles_list = searching_space_df['SMILES'].tolist()

results = {}
for s in smiles_list:
    results[s] = has_B_OH_bond(s)
    print(f"{s:8s} -> {has_B_OH_bond(s)}")


count_true = sum(1 for v in results.values() if v is True)
print(count_true)
