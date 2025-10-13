import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from rdkit import Chem

additive_df = pd.DataFrame(pd.read_excel('./V3/processed_data/20251011_additives_all_V2.xlsx'))
all_smiles = additive_df['SMILES'].tolist()
all_additives = additive_df['additives_abbr'].tolist()
additive_list = []
for additives in all_additives:
    all_additives = additives.split(', ')
    additive_list += all_additives

additive_list = list(set(additive_list))
additive_smiles_pair = {}
for additive in additive_list:
    if additive in ['(C6H3F)O2B(C6H3F2)', '(C6H3F)O2B(C7H4F3)', '(C6H3F)O2B(C8H3F6)', '(C6F4)O2B(C6H4F)', '(C6F4)O2B(C6H3F2)', '(C6F4)O2B(C6F5)', '(C6F4)O2B(C7H3F6)', '(C6F4)O2B(C8H3F6)', '(C6F12)O2B(C6H5)', '(C6F12)O2B(C6H3F2)', '(C6F12)O2B(C6F5)', '(C3HF6O)2B(C6H5)', '(C3HF6O)2B(C6H3F2)', '(C3HF6O)2B(C6F5)']:
        select_df = additive_df.loc[additive_df['idx'] == 306]
    elif additive == 'B(OPh)3':
        select_df = additive_df.loc[additive_df['idx'] == 706]
    elif additive == 'BNNFs':
        select_df = additive_df.loc[additive_df['idx'] == 563]
    elif additive == 'BNNFs':
        select_df = additive_df.loc[additive_df['idx'] == 458]
    elif additive == 'Boroxines':
        SMILES = 'nan'
    elif additive == 'Na[B(hfip)4]':
        select_df = additive_df.loc[additive_df['idx'] == 544]
    elif additive == 'B(HFIP)3':
        select_df = additive_df.loc[additive_df['idx'] == 553]
    elif additive == 'K2B4O5(OH)4':
        select_df = additive_df.loc[additive_df['idx'] == 542]
    else:
        select_df = additive_df.loc[additive_df['additives_abbr'].str.contains(additive)]

    select_additives =  select_df['additives_abbr'].tolist()[0]
    select_smiles =  select_df['SMILES'].tolist()[0]

    if ', ' in select_additives:
        select_additives = select_additives.split(', ')
        if str(select_smiles) == 'nan':
            SMILES = 'nan'
        else:
            select_smiles = select_smiles.split('; ')
    else:
        select_additives = [select_additives]
        select_smiles = [select_smiles]
    
    if str(select_smiles) == 'nan':
        SMILES = 'nan'
    elif additive == 'BN':
        SMILES = 'nan'
    elif additive == 'Li[BScB]':
        SMILES = 'O=C1C2=CC=CC=C2O[B-](OC3=C4C=CC=C3)(OC4=O)O1.[Li+]'
    elif additive == 'LBS':
        SMILES = 'nan'
    elif additive == 'TPF':
        SMILES = 'FC1=C(C(=C(C(=C1B(C1=C(C(=C(C(=C1F)F)F)F)F)C1=C(C(=C(C(=C1F)F)F)F)F)F)F)F)F'
    elif additive == 'Boroxines':
        SMILES = 'nan'
    elif additive == 'THFB':
        SMILES = '[B-](OC(C(F)(F)F)C(F)(F)F)(OC(C(F)(F)F)C(F)(F)F)OC(C(F)(F)F)C(F)(F)F'
    elif additive == 'TEB':
        SMILES == 'B(OCC)(OCC)OCC  '
    else:
        if len(select_additives) == 1:
            SMILES = select_smiles
        else:
            index = select_additives.index(additive)
            SMILES = select_smiles[index]

    additive_smiles_pair[additive] = SMILES

print(len(additive_smiles_pair))

hybridization_dict = {}
for additive, smiles in additive_smiles_pair.items():
    if type(smiles) == list:
        smiles = smiles[0]

    smiles = str(smiles)
    if smiles == 'nan':
        continue
    elif smiles == '[nan]':
        continue
    elif smiles == '':
        continue

    mol = Chem.MolFromSmiles(smiles)

    if smiles == 'CCC[N+]1(F[B-](OC2=O)(OC2=O)F)CCCCC1':
        hybridization_dict[additive] = ['SP3']
    else:
        hybridization_list = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != 'B':
                continue
            hybridization_list.append(str(atom.GetHybridization()))
        
        # hybridization_dict[additive] = hybridization_list
    
    hybridization_dict[additive] = {
            'smiles': smiles,
            'hybridization': hybridization_list
        }

print(len(hybridization_dict))

# print('Number of boron-containing SMILES:', (len(smiles_list)))
# print('Number of boron-containing additives:', (len(additive_list)))
# print(len(hybridization_dict))


functional_groups = {
    "hydroxyl (-OH)": "[OX2H]",                            # 醇羟基
    "carbonyl (C=O)": "[CX3]=[OX1]",                      # 羰基
    "carboxyl (-COOH)": "C(=O)[OX2H1]",                   # 羧基
    "amine (-NH2)": "[NX3;H2]",                           # 胺基
    "amide (-CONH2)": "C(=O)N",                           # 酰胺
    "ester (-COOR)": "C(=O)O[C,H]",                       # 酯
    "ether (-O-)": "C-O-C",                               # 醚
    "nitro (-NO2)": "[NX3](=O)=O",                        # 硝基
    "halide (C-X)": "[C][F,Cl,Br,I]",                     # 卤代烃
    "alkene (C=C)": "C=C",                                # 烯烃
    "alkyne (C#C)": "C#C",                                # 炔烃
    "benzene ring": "c1ccccc1",                           # 苯环
    "boronate (B-O)": "B-O",                              # 硼氧键
}

smiles_list = []
for k, v in hybridization_dict.items():
    smiles = v['smiles']
    smiles_list.append(smiles)

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    found_groups = []
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            found_groups.append(name)
    print(f"SMILES: {smi}")
    print(f"  官能团: {found_groups if found_groups else '无匹配'}\n")

