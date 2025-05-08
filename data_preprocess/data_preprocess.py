import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def preprocess_papers_data():
    """Preprocess the papers label data.

    Args: 
        None.

    Returns:
        final_label_df: the unified papers data in DataFrame format.

    """
    labeled_paper_data_df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/final_papers_label.csv'))
    label_data_df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/label_data_stage4.csv'))

    cids = labeled_paper_data_df['cid'].values.tolist()
    cid_list = []
    doi_list = []
    formulas = []
    smiles = []
    fingerprints = []
    topologicals = []
    weights = []
    heavy_atoms = []
    types = []
    labels = []
    literature_types = []
    for cid in tqdm(set(cids), desc='Preprocessing the papers label data...'):
        dois = labeled_paper_data_df.loc[labeled_paper_data_df['cid'] == cid, 'literature_id'].values

        for doi in dois:
            label_df = labeled_paper_data_df[labeled_paper_data_df['literature_id'] == doi]
            type_value = label_df.loc[label_df['cid'] == cid, 'type'].values[0]
            label_value = label_df.loc[label_df['cid'] == cid, 'label'].values[0]
            select_df = label_data_df[label_data_df['doi'] == doi]
            if select_df.shape[0] == 0:
                select_df = label_data_df[label_data_df['literatures'] == doi]

            formula = select_df.loc[select_df['cid'] == cid, 'formula'].values[0]
            smile = select_df.loc[select_df['cid'] == cid, 'SMILES'].values[0]
            fingerprint = select_df.loc[select_df['cid'] == cid, 'fingerprint'].values[0]
            topological = select_df.loc[select_df['cid'] == cid, 'topological'].values[0]
            weight = select_df.loc[select_df['cid'] == cid, 'weight'].values[0]
            heavy_atom = select_df.loc[select_df['cid'] == cid, 'heavy_atom'].values[0]

            cid_list.append(cid)
            doi_list.append(doi)
            formulas.append(formula)
            smiles.append(smile)
            fingerprints.append(fingerprint)
            topologicals.append(topological)
            weights.append(weight)
            heavy_atoms.append(heavy_atom)
            types.append(type_value)
            labels.append(label_value)
            literature_types.append('paper')

    final_label_df = pd.DataFrame()
    final_label_df['cid'] = cid_list
    final_label_df['doi'] = doi_list
    final_label_df['literature_type'] = literature_types
    final_label_df['formula'] = formulas
    final_label_df['smile'] = smiles
    final_label_df['fingerprint'] = fingerprints
    final_label_df['topological'] = topologicals
    final_label_df['weight'] = weights
    final_label_df['heavy_atom'] = heavy_atoms
    final_label_df['type'] = types
    final_label_df['label'] = labels

    return final_label_df

def preprocess_patents_data():
    """Preprocess the patents label data.

    Args: 
        None.

    Returns:
        final_label_df: the unified patents data in DataFrame format.

    """
    labeled_patent_data_df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/final_patents_label.csv'))
    label_data_df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/label_data_stage4.csv'))

    cids = labeled_patent_data_df['cid'].values.tolist()
    cid_list = []
    doi_list = []
    formulas = []
    smiles = []
    fingerprints = []
    topologicals = []
    weights = []
    heavy_atoms = []
    types = []
    labels = []
    literature_types = []
    for cid in tqdm(set(cids), desc='Preprocessing the patents label data...'):
        dois = labeled_patent_data_df.loc[labeled_patent_data_df['cid'] == cid, 'literature_id'].values

        for doi in dois:
            label_df = labeled_patent_data_df[labeled_patent_data_df['literature_id'] == doi]
            type_value = label_df.loc[label_df['cid'] == cid, 'type'].values[0]
            label_value = label_df.loc[label_df['cid'] == cid, 'label'].values[0]

            if type_value == 'Other' and label_value == '-1':# drop the error labels
                continue

            doi_numbers = label_data_df['doi'].values
            doi_map = {}
            for doi_number in doi_numbers:
                new_doi = str(doi_number).replace('-','')
                doi_map[new_doi] = str(doi_number)
            
            doi = doi_map[doi]
            select_df = label_data_df[label_data_df['doi'] == doi]

            formula = select_df.loc[select_df['cid'] == int(cid), 'formula'].values[0]
            smile = select_df.loc[select_df['cid'] == int(cid), 'SMILES'].values[0]
            fingerprint = select_df.loc[select_df['cid'] == int(cid), 'fingerprint'].values[0]
            topological = select_df.loc[select_df['cid'] == int(cid), 'topological'].values[0]
            weight = select_df.loc[select_df['cid'] == int(cid), 'weight'].values[0]
            heavy_atom = select_df.loc[select_df['cid'] == int(cid), 'heavy_atom'].values[0]

            cid_list.append(cid)
            doi_list.append(doi)
            formulas.append(formula)
            smiles.append(smile)
            fingerprints.append(fingerprint)
            topologicals.append(topological)
            weights.append(weight)
            heavy_atoms.append(heavy_atom)
            types.append(type_value)
            labels.append(label_value)
            literature_types.append('patent')

    final_label_df = pd.DataFrame()
    final_label_df['cid'] = cid_list
    final_label_df['doi'] = doi_list
    final_label_df['literature_type'] = literature_types
    final_label_df['formula'] = formulas
    final_label_df['smile'] = smiles
    final_label_df['fingerprint'] = fingerprints
    final_label_df['topological'] = topologicals
    final_label_df['weight'] = weights
    final_label_df['heavy_atom'] = heavy_atoms
    final_label_df['type'] = types
    final_label_df['label'] = labels

    return final_label_df

preprocessed_papers_df = preprocess_papers_data()
preprocessed_patents_df = preprocess_patents_data()
total_labeled_df = pd.concat([preprocessed_papers_df, preprocessed_patents_df], ignore_index=True)
total_labeled_df = total_labeled_df[~((total_labeled_df['type'] == 'Other') & (total_labeled_df['label'] == '-1'))]
total_labeled_df = total_labeled_df[~(total_labeled_df['type'] == '-1')]
total_labeled_df.to_csv('./PubChem/processed_data/final_labeled_data.csv', index=False)# save the final labeled data

# construct labeled data
cid_list = set(total_labeled_df['cid'].values.tolist())
cids = []
smiles = []
labels = []
for cid in tqdm(cid_list, desc='Constructing labeled data...'):
    sub_labels_df = total_labeled_df.loc[total_labeled_df['cid'] == cid, 'label'].values.tolist()
    if 1 in sub_labels_df:
        labels.append(1)
    else:
        labels.append(0)
    cids.append(cid)
    smiles.append(total_labeled_df.loc[total_labeled_df['cid'] == cid, 'smile'].values[0])

labeled_data_df = pd.DataFrame()
labeled_data_df['cid'] = cids
labeled_data_df['smiles'] = smiles
labeled_data_df['labels'] = labels
labeled_data_df.to_csv('./data/labeled_data.csv', index=False)# save the labeled data for model training

# construct unlabeled data
searching_space_df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/searching_space_data.csv'))
unlabeled_cid_list = searching_space_df['cid'].values.tolist()
un_cids = []
un_smiles = []
for unlabeled_cid in tqdm(unlabeled_cid_list, desc='Constructing unlabeled data...'):
    if unlabeled_cid in cid_list:
        continue

    un_cids.append(unlabeled_cid)
    un_smiles.append(searching_space_df.loc[searching_space_df['cid'] == unlabeled_cid, 'SMILES'].values[0])

unlabeled_data_df = pd.DataFrame()
unlabeled_data_df['cid'] = un_cids
unlabeled_data_df['smiles'] = un_smiles
unlabeled_data_df.to_csv('./data/unlabeled_data.csv', index=False)# save the labeled data for model training