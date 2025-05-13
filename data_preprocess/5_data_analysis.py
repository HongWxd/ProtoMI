import os 
import re
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# load the data
label_data_df = pd.DataFrame(pd.read_csv('./data/labeled_data.csv'))# analysis the labeled data
candidates_data_df = pd.DataFrame(pd.read_csv('./data/searching_space_data.csv'))# analysis the searching space data

# how many different compounds 
cid_count = len(set(label_data_df['cid'].values))
candidates_cid_count = len(set(candidates_data_df['cid'].values))
print('total compound number:', cid_count)
print('total candidates compound number:', candidates_cid_count)

# heavy atom number range
heavy_atom_count_list = label_data_df['heavy_atom'].values.tolist()
max_atom_number = max(heavy_atom_count_list)
min_atom_number = min(heavy_atom_count_list)
print('max heavy atom number:', max_atom_number, 'min heavy atom number:', min_atom_number)

# weight range for the dataset
weight_atom_count_list = label_data_df['weight'].values.tolist()
max_weight_number = max(weight_atom_count_list)
min_weight_number = min(weight_atom_count_list)
print('max weight number:', max_weight_number, 'min weight number:', min_weight_number)

# topological range for the dataset
topological_atom_count_list = label_data_df['topological'].values.tolist()
max_topological_number = max(topological_atom_count_list)
min_topological_number = min(topological_atom_count_list)
print('max topological number:', max_topological_number, 'min topological number:', min_topological_number)

# how many elements and atoms number are included for each compound in the dataset
def parse_molecular_formula(formula):
    atom_count = defaultdict(int)

    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    for atom, count in matches:
        count = int(count) if count else 1
        atom_count[atom] += count
        
    return atom_count

formula_list = set(label_data_df['formula'].values)
atoms_count = []
elements_count = []
for formula in formula_list:
    atom_count = parse_molecular_formula(formula)

    total_atoms = sum(atom_count.values())
    atoms_count.append(total_atoms)
    elements = atom_count.keys()
    for element in elements:
        if element in elements_count:
            continue
        else:
            elements_count.append(element)

max_atom_count = max(atoms_count)
min_atom_count = min(atoms_count)
print('max atom number:', max_atom_count, 'min atom number:', min_atom_count)
print('reported compounds elements:', len(elements_count))

# analysis the type of compounds
def hotspot_plot(plot_type, label_data_df, types):
    type_list = []
    type_len = []
    for key, values in types.items():
        values = set(values)
        type_list = [i for i in values]
        type_len.append(len(values))
        if len(values) > 6:
            print(key, values)
    max_value = max(type_len)

    type_hotspot = []
    labeled_keys = []
    for key, values in types.items():
        libs, mibs, zibs, sibs, lmbs, others = 0,0,0,0,0,0
        values = set(values)
        if len(values) == max_value:
            smile = label_data_df.loc[label_data_df['cid'] == key, 'smile'].values[0]
            type_df = label_data_df.loc[label_data_df['cid'] == key, 'type'].values
            for type_v in type_df:
                try:
                    sub_types = type_v.split(',')
                except:
                    sub_types = 0

                if len(sub_types) != 0:
                    type_v = sub_types
                
                for tyv in type_v:
                    if 'LIB' in tyv:
                        libs = libs + 1
                    elif 'MIB' in tyv:
                        mibs = mibs + 1
                    elif 'ZIB' in tyv:
                        zibs = zibs + 1
                    elif 'SIB' in tyv:
                        sibs = sibs + 1
                    elif 'LMB' in tyv:
                        lmbs = lmbs + 1
                    elif 'Other' in tyv:
                        others = others + 1
                    else:
                        continue
        else:
            continue
        sub_list = [libs, mibs, zibs, sibs, lmbs, others]
        labeled_keys.append(key)
        type_hotspot.append(sub_list)

    # print(labeled_keys)
    type_hotspot_df = pd.DataFrame(type_hotspot, columns=['LIB', 'MIB', 'ZIB', 'SIB', 'LMB', 'Other'])
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(type_hotspot_df), columns=type_hotspot_df.columns)

    if plot_type == 'frequently':
        index = df_normalized[df_normalized > 0.3].stack().index
        index_list = []
        for i in index:
            i = i[0]
            index_list.append(i)
        index_list = set(index_list)

        labeled_data = []
        hotspot_fomulas = []
        for ii in index_list:
            row = (df_normalized.loc[ii].values).tolist()
            compound_key = labeled_keys[ii]
            formula = label_data_df.loc[label_data_df['cid'] == int(compound_key), 'formula'].values[0]

            hotspot_fomulas.append(formula)
            labeled_data.append(row)

        df_hotspot = pd.DataFrame(labeled_data, columns=['LIB', 'MIB', 'ZIB', 'SIB', 'LMB', 'Other'])
        df_normalized = df_hotspot
        row_names = hotspot_fomulas

    elif plot_type == 'all':
        hotspot_fomulas = []
        for i in labeled_keys:
            formula = label_data_df.loc[label_data_df['cid'] == int(i), 'formula'].values[0]
            hotspot_fomulas.append(formula)
            row_names = hotspot_fomulas

    plt.figure(figsize=(9, 5))
    ax = sns.heatmap(df_normalized, annot=True)
    ax.set_yticklabels(row_names, rotation=0, ha='right')
    plt.tight_layout()
    plt.savefig(f'./figs/hotspot_plot_{plot_type}_type.png', dpi=600)

types = {}
for cid in set(label_data_df['cid'].values):
    selected_df = label_data_df[label_data_df['cid'] == cid]
    type_values = selected_df.loc[selected_df['cid'] == cid, 'type'].values

    for type_ in type_values:
        print(type_)
        type_ = type_.split('[')[1].split(']')[0]
        
        print(cid, type_)
        if ',' in type_:
            type_value = type_.split(',')
            print(cid, type_value)
            for type__ in type_value:
                if cid not in types:
                    types[cid] = []
                types[cid].append(type__)
        else:
            if cid not in types:
                types[cid] = []
            types[cid].append(type_)
        
print('compounds have label ratio:', (len(types) / candidates_cid_count) * 100, '%')
hotspot_plot('frequently', label_data_df, types)
# hotspot_plot('all', label_data_df, types)

# analysis the literature ratio



# analysis the label ratio


