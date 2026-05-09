import requests
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from tqdm import tqdm
from rdkit.Contrib.SA_Score import sascorer
from utils.tools import collect_mol_info_by_ids

class PostScreening():
    def __init__(self, args, predict_labels_df):
        self.save_molecules = args.save_molecules
        self.post_screening_output_path = args.post_screening_output_path
        self.method = args.method
        self.EMA = args.EMA
        self.use_decor_loss = args.use_decor_loss
        self.use_topk = args.use_topk

        with open(args.additive_json_path, 'r') as f:
            self.positive_data = json.load(f)

        self.prototypes_data = pd.read_csv(args.best_prototype_path)
        self.searching_space = pd.read_csv(args.searching_space_path)
        self.predict_labels = predict_labels_df

        self.labile_h_smarts = [
            "[OX2H]",       # 羟基–OH
            "[NX3H]",       # 胺–NH, NH2
            "[NX4+H]",      # 存在质子化的氨基
            "[SH]",         # 硫醇–SH
            "[CX3](=O)[OX2H]" # 羧酸–COOH
        ]

    
    def load_all_samples(self):
        # get all predict samples
        labels = np.unique(self.predict_labels['label'].values.tolist())
        predict_samples_data = pd.DataFrame()
        for label in labels:
            samples_id = self.predict_labels.loc[self.predict_labels['label'] == label, 'id'].values.tolist()
            samples = self.searching_space.loc[self.searching_space['cid'].isin(samples_id)]
            label_value = len(samples) * [label]
            samples_df = samples.copy()
            samples_df['label'] = label_value
            predict_samples_data = pd.concat([predict_samples_data, samples_df], axis=0)

        print('original predict samples:', len(predict_samples_data))
        
        return predict_samples_data
    

    def calculate_boundary(self):
        pos_smiles = [v['smiles'] for k, v in self.positive_data.items()]
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
        # print(f'Max positive MW: {max_pos_mw}, Min positive MW: {min_pos_mw}')
        # print(f'Max positive logP: {max_pos_logp}, Min positive logP: {min_pos_logp}')
        # print(f'Max positive SAScore: {max_pos_SAScore}, Min positive SAScore: {min_pos_SAScore}')

        return max_pos_mw, min_pos_mw, max_pos_logp, min_pos_logp, max_pos_SAScore, min_pos_SAScore
    

    def filter_by_CID(self, predict_samples_data):
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

        return filtered_samples_df
    

    def filter_by_molecular_weight(self, samples_df, min_pos_mw, max_pos_mw):
        # filter the predicted samples based on molecular weight: within the range of positive samples
        filtered_samples = []
        for idx, row in samples_df.iterrows():
            smiles = row['SMILES']
            mw = row['weight']
            if min_pos_mw <= mw <= max_pos_mw:
                filtered_samples.append(row)
        filtered_samples_df = pd.DataFrame(filtered_samples)
        print(f'After filtering by molecular weight: {len(filtered_samples_df)}')
        
        return filtered_samples_df
    

    def filter_by_SAScore(self, samples_df):
        # filter by SA Score
        filtered_samples = []
        all_scs = []
        for idx, row in samples_df.iterrows():
            smiles = row['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            SAScore = sascorer.calculateScore(mol)
            all_scs.append(SAScore)
            if SAScore < 4:
                filtered_samples.append(row)
        filtered_samples_df = pd.DataFrame(filtered_samples)

        print(f'After filtering by SA Score: {len(filtered_samples_df)}')

        return filtered_samples_df
    

    def has_labile_h(self, smiles):
            mol = Chem.MolFromSmiles(smiles)
            for smarts in self.labile_h_smarts:
                patt = Chem.MolFromSmarts(smarts)
                if mol.HasSubstructMatch(patt):
                    return True
            return False


    def filter_by_labile_H(self, samples_df):
        # filter by labile H
        filtered_samples = []
        for idx, row in samples_df.iterrows():
            smiles = row['SMILES']
            has = self.has_labile_h(smiles)

            if has:
                continue
            else:
                filtered_samples.append(row)
        filtered_samples_df = pd.DataFrame(filtered_samples)

        print(f'After filtering by labile H: {len(filtered_samples_df)}')

        return filtered_samples_df
    
    
    def is_commercial(self, smiles):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/synonyms/JSON"
        res = requests.get(url)

        if res.status_code != 200:
            return False

        data = res.json()
        synos = data["InformationList"]["Information"][0].get("Synonym", [])

        has_cas = any("-" in s and s.replace("-", "").isdigit() for s in synos)

        return has_cas
    

    def filter_by_commercial(self, samples_df):
        # filter by commercial availability
        filtered_samples = []
        for i, (idx, row) in zip(tqdm(range(len(samples_df))), samples_df.iterrows()):
            smiles = row['SMILES']
            commercial = self.is_commercial(smiles)

            if commercial:
                filtered_samples.append(row)
            else:
                continue

        filtered_samples_df = pd.DataFrame(filtered_samples)
        print(f'After filtering by commercial: {len(filtered_samples_df)}')

        return filtered_samples_df


    def filter(self, emb_unl, unl_ids):
        predict_samples_data = self.load_all_samples()
        max_pos_mw, min_pos_mw, max_pos_logp, min_pos_logp, max_pos_SAScore, min_pos_SAScore = self.calculate_boundary()

        filtered_samples_df = self.filter_by_CID(predict_samples_data)
        filtered_samples_df = self.filter_by_molecular_weight(filtered_samples_df, min_pos_mw, max_pos_mw)
        filtered_samples_df = self.filter_by_SAScore(filtered_samples_df)
        filtered_samples_df = self.filter_by_labile_H(filtered_samples_df)
        filtered_samples_df = self.filter_by_commercial(filtered_samples_df)

        remain_ids = set(filtered_samples_df['cid'].tolist())

        keep_indices = [
            i for i, sample_id in enumerate(unl_ids)
            if sample_id in remain_ids
        ]

        filtered_emb_unl = emb_unl[keep_indices]

        filtered_unl_ids = [
            unl_ids[i] for i in keep_indices
        ]

        # print('Original embeddings:', len(unl_ids))
        # print('Filtered embeddings:', len(filtered_unl_ids))

        mol_info_after_post_screen_df = collect_mol_info_by_ids(filtered_unl_ids, self.searching_space)

        if self.save_molecules:
            mol_info_after_post_screen_df.to_csv(self.post_screening_output_path + f'recommendations_after_post_screening_{self.method}_ema_{self.EMA}_decor_{self.use_decor_loss}_topk_{self.use_topk}.csv', index=False)
            
            # save embeddings
            np.save(self.post_screening_output_path + f'embeddings_after_post_screening_{self.method}_ema_{self.EMA}_decor_{self.use_decor_loss}_topk_{self.use_topk}.npy', filtered_emb_unl.cpu().numpy())


        return mol_info_after_post_screen_df

# args = []
# postscreener = PostScreening(args)
# filtered_samples_df = postscreener.filter()
