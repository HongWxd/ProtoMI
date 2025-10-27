import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def select_boron_related_info():
    """Search boron related data in the return of function search_cid_in_papers_and_patents(). 

    Args: 
        None.

    Returns:
        boron_total_df: the uniform DataFrame format data of possibly boron related compounds data.
    """

    path = './PubChem/compounds/'
    files_path = os.listdir(path)
    files = [i for i in files_path if i.endswith('.json')]
    boron_total_df = pd.DataFrame()
    for file in files:
        print(f'Processing file: {file} ......')
        search_type = file.split('.json')[0].split('PubChem_compound_text_')[1]
        if search_type.startswith('boron_records'):
            boron_df = search_in_boron_file(path + file, type='compound')# search the original file obtained by keywords: [boron] in PubChem website
        else:
            boron_df = search_in_additive_electrolyte_files(path + file)# search the original file obtained by keywords: [additive, electrolyte] in PubChem website
        
        boron_total_df = pd.concat([boron_total_df, boron_df], ignore_index=True)
        
    return boron_total_df

def search_in_boron_file(json_path, type):
    """An assistant function for searching boron related data in the original file obtained by keywords: [boron] in PubChem website.

    Args: 
        json_path: the path for original file obtained from PubChem website.
        type: the variable represents whether it is compounds or others.

    Returns:
        boron_total_df: the uniform DataFrame format data of possibly boron related compounds data.
    """

    with open(json_path, 'r') as f:
        data = json.load(f)

    ids = []
    fingerprints = []
    SMILEs = []
    topologicalbs = []
    weights = []
    heavy_atoms = []
    formulas = []
    for value in tqdm(data[f'PC_{type[0].upper()}{type[1:]}s'], desc=f'Preprocessing {json_path} file into dataframe format'):
        id = value['id']['id'][f'{type[0]}id']
        props = value['props']
        count = value['count']
        for prop in props:
            label = prop['urn']['label']
            if label == 'Fingerprint':
                fingerprint = prop['value']['binary']
            elif label == 'SMILES':
                smile = prop['value']['sval']
            elif label == 'Topological':
                topological = prop['value']['fval']
            elif label == 'Weight':
                weight = prop['value']['sval']
            elif label == 'Molecular Formula':
                formula = prop['value']['sval']
        heavy_atom = count['heavy_atom']

        ids.append(int(id))
        fingerprints.append(str(fingerprint))
        SMILEs.append(str(smile))
        topologicalbs.append(float(topological))
        weights.append(float(weight))
        heavy_atoms.append(int(heavy_atom))
        formulas.append(str(formula))

    boron_df = pd.DataFrame()
    boron_df[f'{type[0]}id'] = ids
    boron_df['formula'] = formulas
    boron_df['SMILES'] = SMILEs
    boron_df['fingerprint'] = fingerprints
    boron_df['topological'] = topologicalbs
    boron_df['weight'] = weights
    boron_df['heavy_atom'] = heavy_atoms

    return boron_df

def search_in_additive_electrolyte_files(json_path):
    """An assistant function for searching boron related data in the original file obtained by keywords: [additive, electrolyte] in PubChem website.

    Args: 
        None.

    Returns:
        boron_total_df: the uniform DataFrame format data of possibly boron related compounds data.
    """
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cids = []
    fingerprints = []
    SMILEs = []
    topologicalbs = []
    weights = []
    heavy_atoms = []
    formulas = []
    for compound in tqdm(data, desc=f'Preprocessing {json_path} file into dataframe format'):
        smiles = str(compound['smiles'])
        if 'B' in smiles and 'Br' not in smiles and 'Ba' not in smiles and 'Bi' not in smiles:
            cids.append(int(compound['cid']))
            fingerprints.append('')
            SMILEs.append(smiles)
            topologicalbs.append(float(0))
            weights.append(float(compound['mw']))
            heavy_atoms.append(int(compound['heavycnt']))
            formulas.append(str(compound['mf']))
    
    boron_df = pd.DataFrame()
    boron_df['cid'] = cids
    boron_df['formula'] = formulas
    boron_df['SMILES'] = SMILEs
    boron_df['fingerprint'] = fingerprints
    boron_df['topological'] = topologicalbs
    boron_df['weight'] = weights
    boron_df['heavy_atom'] = heavy_atoms

    return boron_df


# step2: read out the all boron related compounds from json data
boron_df = select_boron_related_info()
boron_df = boron_df.drop_duplicates(subset='cid', keep='first')
boron_df.to_csv(f'./PubChem/processed_data/searching_space_data_V2.csv', index=False)# get the whole searching sapce
print('The searching space compounds information is save at: ./PubChem/processed_data/searching_space_data_V2.csv')
