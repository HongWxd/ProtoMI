import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def search_cid_in_papers_and_patents():
    """Search possible battery related papers and patents in the PubChem website, 
    we use keywords: BORON, ELECTROLYTE, ADDITIVES, to download the original files.

    Args: 
        None.

    Returns:
        battery_cids_final: the uniform dictionary data of possibly battery related compounds, 
        collected by keywords: BATTER, ELECTROLYT, ADDITIVE, SEI, ANODE, CATHODE, ELECTRODE both for papers and patents data.

    """

    papers_df = pd.DataFrame(pd.read_csv('./PubChem/literatures/PubChem_pubmed_text_boron.csv'))# load the papers data
    patents_df = pd.DataFrame(pd.read_csv('./PubChem/literatures/PubChem_patent_text_boron.csv'))# load the patents data

    papers_title_list = papers_df['articletitle'].values.tolist()
    papers_abs_list = papers_df['articleabstract'].values.tolist()
    patents_title_list = patents_df['title'].values.tolist()
    patents_abs_list = patents_df['abstract'].values.tolist()

    battery_cids = {}
    battery_pids = {}
    battery_dois = {}
    for title, abs in zip(tqdm(papers_title_list, desc='Searching papers related to keyword boron'), papers_abs_list):# search the related cids from papers material
        if 'BATTER' in str(title).upper() or 'BATTER' in str(abs).upper() or 'ELECTROLYT' in str(title).upper() or 'ELECTROLYT' in str(abs).upper() or 'ADDITIVE' in str(title).upper() or 'ADDITIVE' in str(abs).upper() or ' SEI' in str(title).upper() or ' SEI' in str(abs).upper() or 'ANODE' in str(title).upper() or 'ANODE' in str(abs).upper() or 'CATHODE' in str(title).upper() or 'CATHODE' in str(abs).upper() or 'ELECTRODE' in str(title).upper() or 'ELECTRODE' in str(abs).upper():
            cids = papers_df.loc[papers_df['articletitle'] == title, 'cids'].values[0]
            pids = papers_df.loc[papers_df['articletitle'] == title, ' pmid'].values[0]
            dois = papers_df.loc[papers_df['articletitle'] == title, 'doi'].values[0]

            battery_cids[title] = cids
            battery_pids[title] = pids
            battery_dois[title] = dois
    for title, abs in zip(tqdm(patents_title_list, desc='Searching patents related to keyword boron'), patents_abs_list):# search the related cids from patents material
        if 'BATTER' in str(title).upper() or 'BATTER' in str(abs).upper() or 'ELECTROLYT' in str(title).upper() or 'ELECTROLYT' in str(abs).upper() or 'ADDITIVE' in str(title).upper() or 'ADDITIVE' in str(abs).upper() or ' SEI' in str(title).upper() or ' SEI' in str(abs).upper() or 'ANODE' in str(title).upper() or 'ANODE' in str(abs).upper() or 'CATHODE' in str(title).upper() or 'CATHODE' in str(abs).upper() or 'ELECTRODE' in str(title).upper() or 'ELECTRODE' in str(abs).upper():
            cids = patents_df.loc[patents_df['title'] == title, 'cids'].values
            pids = patents_df.loc[patents_df['title'] == title, ' publicationnumber'].values[0]
            if len(cids) == 0:
                continue
            else:
                cids = cids[0]

            battery_cids[title] = cids
            battery_pids[title] = pids
    
    patents_df = pd.DataFrame(pd.read_csv('./PubChem/literatures/PubChem_patent_text_additive.csv'))# load the patents data
    patents_title_list = patents_df['title'].values.tolist()
    patents_abs_list = patents_df['abstract'].values.tolist()

    for title, abs in zip(tqdm(patents_title_list, desc='Searching patents related to keyword additive'), patents_abs_list):# search the related cids from patents material
        if 'BATTER' in str(title).upper() or 'BATTER' in str(abs).upper() or 'ELECTROLYTE' in str(title).upper() or 'ELECTROLYTE' in str(abs).upper() or ' SEI' in str(title).upper() or ' SEI' in str(abs).upper() or 'ANODE' in str(title).upper() or 'ANODE' in str(abs).upper() or 'CATHODE' in str(title).upper() or 'CATHODE' in str(abs).upper() or 'ELECTRODE' in str(title).upper() or 'ELECTRODE' in str(abs).upper():
            cids = patents_df.loc[patents_df['title'] == title, 'cids'].values
            pids = patents_df.loc[patents_df['title'] == title, ' publicationnumber'].values[0]
            if len(cids) == 0:
                continue
            else:
                cids = cids[0]
            battery_cids[title] = cids
            battery_pids[title] = pids
    
    papers_df = pd.DataFrame(pd.read_csv('./PubChem/literatures/PubChem_pubmed_text_electrolyte.csv'))# load the papers data
    patents_df = pd.DataFrame(pd.read_csv('./PubChem/literatures/PubChem_patent_text_electrolyte.csv'))# load the patents data

    papers_title_list = papers_df['articletitle'].values.tolist()
    papers_abs_list = papers_df['articleabstract'].values.tolist()
    patents_title_list = patents_df['title'].values.tolist()
    patents_abs_list = patents_df['abstract'].values.tolist()

    for title, abs in zip(tqdm(papers_title_list, desc='Searching papers related to keyword electrolyte'), papers_abs_list):# search the related cids from papers material
        if 'ELECTROLYT' in str(title).upper() or 'ELECTROLYT' in str(abs).upper() or 'BATTER' in str(title).upper() or 'BATTER' in str(abs).upper() or 'ADDITIVE' in str(title).upper() or 'ADDITIVE' in str(abs).upper() or ' SEI' in str(title).upper() or ' SEI' in str(abs).upper() or 'ANODE' in str(title).upper() or 'ANODE' in str(abs).upper() or 'CATHODE' in str(title).upper() or 'CATHODE' in str(abs).upper() or 'ELECTRODE' in str(title).upper() or 'ELECTRODE' in str(abs).upper():
            cids = papers_df.loc[papers_df['articletitle'] == title, 'cids'].values[0]
            pids = papers_df.loc[papers_df['articletitle'] == title, ' pmid'].values[0]
            dois = papers_df.loc[papers_df['articletitle'] == title, 'doi'].values[0]
            battery_cids[title] = cids
            battery_pids[title] = pids
            battery_dois[title] = dois
    for title, abs in zip(tqdm(patents_title_list, desc='Searching patents related to keyword electrolyte'), patents_abs_list):# search the related cids from patents material
        if 'ELECTROLYT' in str(title).upper() or 'ELECTROLYT' in str(abs).upper() or 'BATTER' in str(title).upper() or 'BATTER' in str(abs).upper() or 'ADDITIVE' in str(title).upper() or 'ADDITIVE' in str(abs).upper() or ' SEI' in str(title).upper() or ' SEI' in str(abs).upper() or 'ANODE' in str(title).upper() or 'ANODE' in str(abs).upper() or 'CATHODE' in str(title).upper() or 'CATHODE' in str(abs).upper() or 'ELECTRODE' in str(title).upper() or 'ELECTRODE' in str(abs).upper():
            cids = patents_df.loc[patents_df['title'] == title, 'cids'].values
            pids = patents_df.loc[patents_df['title'] == title, ' publicationnumber'].values
            if len(cids) == 0:
                continue
            else:
                cids = cids[0]
                pids = pids[0]
            battery_cids[title] = cids
            battery_pids[title] = pids

    battery_cids_final = pd.DataFrame()
    keys = []
    literatures = []
    pids = []
    dois = []
    for key, value in battery_cids.items():# reform the dictionary into dataframe format
        key = str(key)# literatures
        value = str(value)# cids
        pid = battery_pids[key]# pid: includes paper and patent
        try:
            doi = battery_dois[key]
        except:
            doi = battery_pids[key]# patents don't have doi number
        
        if value == 'nan' or value == '':
            continue

        if '|' in value:
            values = value.split('|')
        else:
            values = [value]

        for cid in values:
            keys.append(cid)
            literatures.append(key)
            pids.append(pid)
            dois.append(doi)
    
    battery_cids_final['cid'] = keys
    battery_cids_final['literature'] = literatures
    battery_cids_final['pids'] = pids
    battery_cids_final['dois'] = dois
    
    return battery_cids_final

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
        search_type = file.split('.json')[0].split('PubChem_compound_text_')[1]
        if search_type.startswith('boron'):
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

def get_label_data(boron_df, battery_cids_final):
    """After get all possible boron related compounds data, use this function to gather all these data into an uniform format.

    Args: 
        boron_df: all possible boron related compounds data.
        battery_cids_final: all possible battery related compounds data.

    Returns:
        battery_related_boron_dict: the uniform dictionary data of possibly boron related compounds data need to be labeled.
    """

    cids = []
    pids = []
    dois = []
    fingerprints = []
    SMILEs = []
    topologicalbs = []
    weights = []
    heavy_atoms = []
    literatures = []
    formulas = []
    boron_related_cids = battery_cids_final['cid'].values.tolist()
    boron_related_literatures = battery_cids_final['literature'].values.tolist()
    boron_related_pids = battery_cids_final['pids'].values.tolist()
    boron_related_dois = battery_cids_final['dois'].values.tolist()
    for cid, value, pid, doi in zip(tqdm(boron_related_cids, desc='Reading out all boron related compounds to label'), boron_related_literatures, boron_related_pids, boron_related_dois):
        if cid == '':
            continue

        if int(cid) == 5462222 or int(cid) == 944 or int(cid) == 2724274 or int(cid) == 5461123 or int(cid) == 6251:
                continue
        
        if int(cid) in boron_df['cid'].values.tolist():
            cids.append(cid)
            pids.append(pid)
            dois.append(doi)
            fingerprints.append(str(boron_df.loc[boron_df['cid'] == int(cid), 'fingerprint'].values[0]))
            SMILEs.append(str(boron_df.loc[boron_df['cid'] == int(cid), 'SMILES'].values[0]))
            topologicalbs.append(float(boron_df.loc[boron_df['cid'] == int(cid), 'topological'].values[0]))
            weights.append(float(boron_df.loc[boron_df['cid'] == int(cid), 'weight'].values[0]))
            heavy_atoms.append(int(boron_df.loc[boron_df['cid'] == int(cid), 'heavy_atom'].values[0]))
            literatures.append(str(value))
            formulas.append(str(boron_df.loc[boron_df['cid'] == int(cid), 'formula'].values[0]))

    battery_related_boron_dict = pd.DataFrame()
    battery_related_boron_dict['cid'] = cids
    battery_related_boron_dict['pid'] = pids
    battery_related_boron_dict['doi'] = dois
    battery_related_boron_dict['literatures'] = literatures
    battery_related_boron_dict['formula'] = formulas
    battery_related_boron_dict['SMILES'] = SMILEs
    battery_related_boron_dict['fingerprint'] = fingerprints
    battery_related_boron_dict['topological'] = topologicalbs
    battery_related_boron_dict['weight'] = weights
    battery_related_boron_dict['heavy_atom'] = heavy_atoms

    battery_related_boron_dict = battery_related_boron_dict.sort_values(by='cid', ascending=True)

    return battery_related_boron_dict


# step1: get the cids possibly related to battery from papers and patents
battery_cids_final = search_cid_in_papers_and_patents()
print('The number of compounds related to battery field (boron-containing and other compounds): ', len(set(battery_cids_final['cid'].values.tolist())))

# step2: read out the all boron related compounds from json data
boron_df = select_boron_related_info()
boron_df.to_csv(f'./PubChem/processed_data/searching_space_data.csv', index=False)# get the whole searching sapce
print('The searching space compounds information is save at: ./PubChem/processed_data/searching_space_data.csv')

# step3: make sure the cids are related to boron compounds and save them with corresponding papers or patents
battery_related_boron_dict = get_label_data(boron_df, battery_cids_final)
battery_related_boron_dict.to_csv('./PubChem/processed_data/label_data_stage4.csv', index=False)
print('The number of boron-containing compounds: ', len(set(battery_related_boron_dict['formula'].values.tolist())))
print('And the boron-containing compounds information is save at: ./PubChem/processed_data/label_data_stage4.csv')