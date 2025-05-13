import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

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

def post_selection_by_AI(labeled_data_df):
    smiles = labeled_data_df['smiles'].values.tolist()
    formulas = labeled_data_df['formula'].values.tolist()
    client = OpenAI(api_key="sk-846361ec44554e6dbacc9fc7a103232b", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "假设你是有机电化学领域的专家，"},
                        {"role": "user", "content": f"请根据我给你提供的物质信息为我做下一步判断: 首先，我有一一对应的物质分子式列表和SMILES表达式列表，物质分子式：{smiles}, SMILES表达式：{formulas}"},
                        {"role": "user", "content": "按照以下逻辑帮我做判断："
                        "1. 根据提供的物质分子式和SMILES表达式逐一判断该物质是否是稳态的物质，如果是，保留，如果不是，则同时删除它们对应的分子式和SMILES表达式；"
                        "2. 从全局的角度帮我过滤一遍我提供给你的物质中是否存在具有不同的SMILES表达式或不同的物质分子式，但他们所要描述的物质其实是同一个的情况，"
                        "如果有，帮我删除其中一个，如果没有，保留。"
                        "最后，严格按照列表格式同时返回给我一个过滤后的物质分子式列表和一个SMILES表达式列表"}
                    ],
                    stream=False
                )
    response_content = response.choices[0].message.content
    with open('./data/post_selection.txt', 'w') as file:
        file.write(response_content)
    
def construct_labeled_data(total_labeled_df):
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

    # post selection: use AI to screen the unreasonable compounds in the labeled data
    all_data_df = pd.DataFrame(pd.read_csv('./data/searching_space_data.csv'))
    AI_labeled_data_df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/final_labeled_data.csv'))

    cids = labeled_data_df['cid'].values.tolist()
    formulas = []
    for cid in cids:
        formula = all_data_df.loc[all_data_df['cid'] == float(cid), 'formula'].values[0]
        formulas.append(formula)

    labeled_data_df['formula'] = formulas

    # post_selection_by_AI(labeled_data_df)
    
    with open('./data/post_selection.txt', 'r') as file:
        content = file.read()
    content = content.split('物质分子式列表：\n[')[1].split(',')
    smiles = []
    for smile in content:
        if 'B(C1=CC=CC=C1C2=CC=CC(=C2)C3=CC=CC=C3)(O)O' in smile:
            smiles.append('B(C1=CC=CC=C1C2=CC=CC(=C2)C3=CC=CC=C3)(O)O')
            continue
        smile = smile.split('\n ')[1].split("'")[1].split("'")[0]
        smiles.append(smile)

    smiles = set(smiles)
    new_formulas = []
    new_cids = []
    new_labels = []
    new_fingerprints = []
    new_topologicals = []
    new_weights = []
    new_heavy_atoms = []
    new_types = []
    for smile in smiles:
        new_cids.append(labeled_data_df.loc[labeled_data_df['smiles'] == smile, 'cid'].values[0])
        new_formulas.append(labeled_data_df.loc[labeled_data_df['smiles'] == smile, 'formula'].values[0])
        new_labels.append(labeled_data_df.loc[labeled_data_df['smiles'] == smile, 'labels'].values[0])
        new_fingerprints.append(all_data_df.loc[all_data_df['SMILES'] == smile, 'fingerprint'].values[0])
        new_topologicals.append(all_data_df.loc[all_data_df['SMILES'] == smile, 'topological'].values[0])
        new_weights.append(all_data_df.loc[all_data_df['SMILES'] == smile, 'weight'].values[0])
        new_heavy_atoms.append(all_data_df.loc[all_data_df['SMILES'] == smile, 'heavy_atom'].values[0])
        type_list = list(set(AI_labeled_data_df.loc[AI_labeled_data_df['smile'] == smile, 'type'].values))

        types = []
        for type in type_list:
            if len(type_list) == 1:
                types.append(type)
            else:
                sub_types = type.split(',')              
                for s_type in sub_types:
                    if s_type in types:
                        continue
                    else:
                        types.append(s_type)
        new_types.append(types)

    new_labeled_data_df = pd.DataFrame()
    new_labeled_data_df['cid'] = new_cids
    new_labeled_data_df['formula'] = new_formulas
    new_labeled_data_df['smile'] = list(smiles)
    new_labeled_data_df['fingerprint'] = new_fingerprints
    new_labeled_data_df['topological'] = new_topologicals
    new_labeled_data_df['weight'] = new_weights
    new_labeled_data_df['heavy_atom'] = new_heavy_atoms
    new_labeled_data_df['type'] = new_types
    new_labeled_data_df['label'] = new_labels
    new_labeled_data_df.to_csv('./data/labeled_data.csv', index=False)# save the labeled data for model training

    return new_labeled_data_df

def construct_unlabeled_data(labeled_data_df):
    # construct unlabeled data
    searching_space_df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/searching_space_data.csv'))
    unlabeled_cid_list = searching_space_df['cid'].values.tolist()
    cid_list = set(labeled_data_df['cid'].values.tolist())
    un_cids = []
    un_smiles = []
    un_formulas = []
    un_fingerprints = []
    un_topologicals = []
    un_weights = []
    un_heavy_atoms = []
    for unlabeled_cid in tqdm(unlabeled_cid_list, desc='Constructing unlabeled data...'):
        if unlabeled_cid in cid_list:
            continue

        un_cids.append(unlabeled_cid)
        un_smiles.append(searching_space_df.loc[searching_space_df['cid'] == unlabeled_cid, 'SMILES'].values[0])
        un_formulas.append(searching_space_df.loc[searching_space_df['cid'] == unlabeled_cid, 'formula'].values[0])
        un_fingerprints.append(searching_space_df.loc[searching_space_df['cid'] == unlabeled_cid, 'fingerprint'].values[0])
        un_topologicals.append(searching_space_df.loc[searching_space_df['cid'] == unlabeled_cid, 'topological'].values[0])
        un_weights.append(searching_space_df.loc[searching_space_df['cid'] == unlabeled_cid, 'weight'].values[0])
        un_heavy_atoms.append(searching_space_df.loc[searching_space_df['cid'] == unlabeled_cid, 'heavy_atom'].values[0])

    unlabeled_data_df = pd.DataFrame()
    unlabeled_data_df['cid'] = un_cids
    unlabeled_data_df['formula'] = un_formulas
    unlabeled_data_df['smiles'] = un_smiles
    unlabeled_data_df['fingerprint'] = un_fingerprints
    unlabeled_data_df['topological'] = un_topologicals
    unlabeled_data_df['weight'] = un_weights
    unlabeled_data_df['heavy_atom'] = un_heavy_atoms
    unlabeled_data_df.to_csv('./data/unlabeled_data.csv', index=False)# save the labeled data for model training

# preprocessed_papers_df = preprocess_papers_data()
# preprocessed_patents_df = preprocess_patents_data()
# total_labeled_df = pd.concat([preprocessed_papers_df, preprocessed_patents_df], ignore_index=True)
# total_labeled_df = total_labeled_df[~((total_labeled_df['type'] == 'Other') & (total_labeled_df['label'] == '-1'))]
# total_labeled_df = total_labeled_df[~(total_labeled_df['type'] == '-1')]
# total_labeled_df.to_csv('./PubChem/processed_data/final_labeled_data.csv', index=False)# save the final labeled data
total_labeled_df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/final_labeled_data.csv'))
labeled_data_df = construct_labeled_data(total_labeled_df)
# construct_unlabeled_data(labeled_data_df)
