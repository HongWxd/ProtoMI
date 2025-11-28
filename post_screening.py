import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
import pandas as pd
import json
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
from rdkit.Contrib.SA_Score import sascorer


with open('./V3/processed_data/additives.json', 'r') as f:
    positive_data = json.load(f)

prototypes_data = pd.read_csv('./result_files/proto_table_trial_7.csv')
searching_space = pd.read_csv('./data/searching_space_data_V2.csv')
predict_labels = pd.read_csv('./result_files/predicted_labels.csv')


# get all predict samples
labels = np.unique(predict_labels['label'].values.tolist())
predict_samples_data = pd.DataFrame()
for label in labels:
    samples_id = predict_labels.loc[predict_labels['label'] == label, 'id'].values.tolist()
    samples = searching_space.loc[searching_space['cid'].isin(samples_id)]
    label_value = len(samples) * [label]
    samples_df = samples.copy()
    samples_df['label'] = label_value
    predict_samples_data = pd.concat([predict_samples_data, samples_df], axis=0)

print('original predict samples:', len(predict_samples_data))


# get the molecular weight from all positive samples
pos_smiles = [v['smiles'] for k, v in positive_data.items()]
pos_mw = []
pos_logp = []
pos_SAScore = []
for smiles in pos_smiles:
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    SAScore = sascorer.calculateScore(mol)
    pos_mw.append(mw)
    if logp < -10:
        continue
    else:
        pos_logp.append(logp)
    pos_SAScore.append(SAScore)


max_pos_mw = max(pos_mw)
min_pos_mw = min(pos_mw)
max_pos_logp = max(pos_logp)
min_pos_logp = min(pos_logp)
max_pos_SAScore = max(pos_SAScore)
min_pos_SAScore = min(pos_SAScore)
print(f'Max positive MW: {max_pos_mw}, Min positive MW: {min_pos_mw}')
print(f'Max positive logP: {max_pos_logp}, Min positive logP: {min_pos_logp}')
print(f'Max positive SAScore: {max_pos_SAScore}, Min positive SAScore: {min_pos_SAScore}')



# filter the predicted samples by abnormal cid
filtered_samples = []
for idx, row in predict_samples_data.iterrows():
    cid = row['cid']
    smiles = row['SMILES']
   
    if '.' in smiles:
        continue

    if cid < 100000000:
        filtered_samples.append(row)
filtered_samples_df = pd.DataFrame(filtered_samples)
print(f'After filtering by abnormal cid: {len(filtered_samples_df)}')


# filter the predicted samples based on molecular weight: within the range of positive samples
filtered_samples = []
for idx, row in filtered_samples_df.iterrows():
    smiles = row['SMILES']
    mw = row['weight']
    if min_pos_mw <= mw <= max_pos_mw:
        filtered_samples.append(row)
filtered_samples_df = pd.DataFrame(filtered_samples)
print(f'After filtering by molecular weight: {len(filtered_samples_df)}')


# further filter the predicted samples based on logP: within the range of positive samples
filtered_samples = []
for idx, row in filtered_samples_df.iterrows():
    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(smiles)
    logp = Crippen.MolLogP(mol)
    if min_pos_logp <= logp <= max_pos_logp:
        filtered_samples.append(row)
filtered_samples_df = pd.DataFrame(filtered_samples)
print(f'After filtering by logp: {len(filtered_samples_df)}')


# filter by SA Score
filtered_samples = []
all_scs = []
for idx, row in filtered_samples_df.iterrows():
    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(smiles)
    SAScore = sascorer.calculateScore(mol)
    all_scs.append(SAScore)
    if min_pos_SAScore <=  SAScore < 4:
        filtered_samples.append(row)
filtered_samples_df = pd.DataFrame(filtered_samples)

print(f'After filtering by SA Score: {len(filtered_samples_df)}')


labile_h_smarts = [
    "[OX2H]",       # 羟基–OH
    "[NX3H]",       # 胺–NH, NH2
    "[NX4+H]",      # 存在质子化的氨基
    "[SH]",         # 硫醇–SH
    "[CX3](=O)[OX2H]" # 羧酸–COOH
]

def has_labile_h(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for smarts in labile_h_smarts:
        patt = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(patt):
            return True
    return False

# filter by labile H
filtered_samples = []
all_scs = []
for idx, row in filtered_samples_df.iterrows():
    smiles = row['SMILES']
    has = has_labile_h(smiles)

    if has:
        continue
    else:
        filtered_samples.append(row)
filtered_samples_df = pd.DataFrame(filtered_samples)

print(f'After filtering by labile H: {len(filtered_samples_df)}')
filtered_samples_df_without_commercial = filtered_samples_df.copy()
filtered_samples_df_without_commercial.to_csv('./result_files/filtered_predicted_additives_without_commercial.csv', index=False)


import requests

def is_commercial(smiles):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/synonyms/JSON"
    res = requests.get(url)

    if res.status_code != 200:
        return False

    data = res.json()
    synos = data["InformationList"]["Information"][0].get("Synonym", [])

    # 判断是否有 CAS号 → 通常代表商业可买
    has_cas = any("-" in s and s.replace("-", "").isdigit() for s in synos)

    return has_cas

# filter by commercial availability
filtered_samples = []
all_scs = []
for i, (idx, row) in zip(tqdm(range(len(filtered_samples_df))), filtered_samples_df.iterrows()):
    smiles = row['SMILES']
    commercial = is_commercial(smiles)

    if commercial:
        filtered_samples.append(row)
    else:
        continue

filtered_samples_df = pd.DataFrame(filtered_samples)
print(f'After filtering by commercial: {len(filtered_samples_df)}')

filtered_samples_df.to_csv('./result_files/filtered_predicted_additives.csv', index=False)
