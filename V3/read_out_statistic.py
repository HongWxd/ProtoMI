import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from rdkit import Chem

additive_df = pd.DataFrame(pd.read_excel('./V3/processed_data/20251011_additives_all_V2.xlsx'))
all_smiles = additive_df['SMILES'].tolist()
all_additives = additive_df['additives_abbr'].tolist()
origin_additive_list = []
for additives in all_additives:
    all_additives = additives.split(', ')
    origin_additive_list += all_additives

counter = Counter(origin_additive_list)
# print(counter)
# top = counter.most_common(30)
# labels, values = zip(*top)

# plt.figure(figsize=(10, 6))
# bars = plt.bar(labels, values)
# for bar, value in zip(bars, values):
#     plt.text(
#         bar.get_x() + bar.get_width() / 2, 
#         bar.get_height(),                  
#         str(value),                        
#         ha='center', va='bottom'           
#     )
# plt.xticks(rotation=45, ha='right')
# plt.xlabel("Additives Name")
# plt.ylabel("Reported Time")
# plt.title(f"The top {30} most frequently occurring elements")
# plt.tight_layout()
# plt.savefig('./V3/plots/additive_count.png', dpi=600)

additive_list = list(set(origin_additive_list))
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
    electrolyte_types = select_df['electrolyte_types'].tolist()

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
new_dict = {}
for k, v in hybridization_dict.items():
    smiles = v['smiles']
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES for {k}: {smiles}")
        continue
    found_groups = []
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            found_groups.append(name)
    
    new_dict[k] = {
            'smiles': smiles,
            'hybridization': v['hybridization'],
            'functional_groups': found_groups if found_groups else 'None'
        }

print(len(new_dict))

# # electrolyte type vs year
# year_electrolyte_type_pair = {}
# for additive, values in additive_smiles_pair.items():
#     if additive in ['(C6H3F)O2B(C6H3F2)', '(C6H3F)O2B(C7H4F3)', '(C6H3F)O2B(C8H3F6)', '(C6F4)O2B(C6H4F)', '(C6F4)O2B(C6H3F2)', '(C6F4)O2B(C6F5)', '(C6F4)O2B(C7H3F6)', '(C6F4)O2B(C8H3F6)', '(C6F12)O2B(C6H5)', '(C6F12)O2B(C6H3F2)', '(C6F12)O2B(C6F5)', '(C3HF6O)2B(C6H5)', '(C3HF6O)2B(C6H3F2)', '(C3HF6O)2B(C6F5)']:
#         select_df = additive_df.loc[additive_df['idx'] == 306]
#     elif additive == 'B(OPh)3':
#         select_df = additive_df.loc[additive_df['idx'] == 706]
#     elif additive == 'BNNFs':
#         select_df = additive_df.loc[additive_df['idx'] == 563]
#     elif additive == 'BNNFs':
#         select_df = additive_df.loc[additive_df['idx'] == 458] 
#     elif additive == 'Boroxines':
#         SMILES = 'nan'
#     elif additive == 'Na[B(hfip)4]':
#         select_df = additive_df.loc[additive_df['idx'] == 544]
#     elif additive == 'B(HFIP)3':
#         select_df = additive_df.loc[additive_df['idx'] == 553]
#     elif additive == 'K2B4O5(OH)4':
#         select_df = additive_df.loc[additive_df['idx'] == 542]
#     else:
#         select_df = additive_df.loc[additive_df['additives_abbr'].str.contains(additive)]
#     electrolyte_types = select_df['electrolyte_types'].tolist()
#     years = select_df['years'].tolist()

#     if len(years) > 1:
#         for i, year in enumerate(years):
#             if year in year_electrolyte_type_pair.keys():
#                 year_electrolyte_type_pair[year].append(electrolyte_types[i])
#             else:
#                 year_electrolyte_type_pair[year] = [electrolyte_types[i]]
#     else:
#         if years[0] in year_electrolyte_type_pair.keys():
#             year_electrolyte_type_pair[years[0]].append(electrolyte_types[0])
#         else:
#             year_electrolyte_type_pair[years[0]] = [electrolyte_types[0]]

# for year, types in year_electrolyte_type_pair.items():
#     type_counter = Counter(types)
#     year_electrolyte_type_pair[year] = dict(type_counter)

# df = pd.DataFrame(year_electrolyte_type_pair).T.fillna(0)
# df = df[['Liquid', 'Solid-State', 'Polymer']]
# df = df.sort_index() 

# df_percent = df.div(df.sum(axis=1), axis=0) * 100

# plt.figure(figsize=(12,6))
# df.plot(kind='bar', stacked=True, color=['#4C72B0', '#55A868', '#C44E52'], width=0.8)

# plt.title("Proportion of Electrolyte Types by Year", fontsize=14)
# plt.xlabel("Year", fontsize=12)
# plt.ylabel("Percentage (%)", fontsize=12)
# plt.legend(title="Type", loc='upper left')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('./V3/plots/electrolyte_type_trend.png', dpi=600)


# hybirdization vs electrolyte type
hybridization_electrolyte_type_pair = {}
for additive, values in new_dict.items():
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
    electrolyte_types = select_df['electrolyte_types'].tolist()
    hybridization_types = list(set(values['hybridization']))

    if hybridization_types[0] in hybridization_electrolyte_type_pair.keys():
        if len(electrolyte_types) > 1:
            for etype in electrolyte_types:
                    hybridization_electrolyte_type_pair[hybridization_types[0]].append(etype)
        else:
            hybridization_electrolyte_type_pair[hybridization_types[0]].append(electrolyte_types[0])
    else:
        if len(electrolyte_types) > 1:
            for i, etype in enumerate(electrolyte_types):
                if i < 1:
                    hybridization_electrolyte_type_pair[hybridization_types[0]] = [electrolyte_types[0]]
                else:
                    hybridization_electrolyte_type_pair[hybridization_types[0]].append(etype)
        else:
            hybridization_electrolyte_type_pair[hybridization_types[0]] = [electrolyte_types[0]]
        
print(hybridization_electrolyte_type_pair)

for hybridization, types in hybridization_electrolyte_type_pair.items():
    type_counter = Counter(types)
    hybridization_electrolyte_type_pair[hybridization] = dict(type_counter)

df = pd.DataFrame(hybridization_electrolyte_type_pair).T.fillna(0)
df = df[['Liquid', 'Solid-State', 'Polymer']]
df = df.sort_index() 

df_percent = df.div(df.sum(axis=1), axis=0) * 100

plt.figure(figsize=(12,6))
df.plot(kind='bar', stacked=True, color=['#4C72B0', '#55A868', '#C44E52'], width=0.8)

plt.title("Proportion of Electrolyte Types by Hybridization of Boron Element", fontsize=12)
plt.xlabel("Hybridization Ways", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.legend(title="Type", loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./V3/plots/electrolyte_type_hybridization.png', dpi=600)


