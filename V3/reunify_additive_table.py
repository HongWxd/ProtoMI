import os
import pandas as pd
import numpy as np

additive_df = pd.DataFrame(pd.read_csv('./V3/processed_data/additives_all.csv'))
additives = additive_df['additives'].tolist()
additives_abbr = (additive_df['additives_abbr'].tolist())
idx = additive_df['idx'].tolist()

total_additives_abbr = []
for additive_abbr, additive in zip(additives_abbr, additives):
    if '(C6H3F)O2B(C6H3F2), (C6F4)O2B(C6F5)' in additive_abbr:
        additive_abbr = '(C6H3F)O2B(C6H3F2), (C6F4)O2B(C6F5)'
    
    if len(additive_abbr) > 1:
        sub_additives_abbr_list = additive_abbr.split(', ')
        total_additives_abbr += sub_additives_abbr_list
    else:
        total_additives_abbr.append(additive_abbr)


additive_abbr_single = []
idx_each_additive = []
for additive_abbr in total_additives_abbr:
    for id, additive in zip(idx, additives_abbr):
        sub_additives_abbr_list = []
        if len(additive) > 1:
            sub_additives_abbr_list = additive.split(', ')
            if additive_abbr in sub_additives_abbr_list:
                additive_abbr_single.append(additive_abbr)
                idx_each_additive.append(id)
        else:
            if additive_abbr in [additive]:
                additive_abbr_single.append(additive_abbr)
                idx_each_additive.append(id)

electrolytes_info = []
cathode_info = []
anode_indo = []
cell_type_info = []
for id, additive_name in zip(idx_each_additive, additive_abbr_single):
    electrolytes_info.append(additive_df.loc[additive_df['idx'] == int(id), 'electrolyte'].values[0])
    cathode_info.append(additive_df.loc[additive_df['idx'] == int(id), 'cathode'].values[0])
    anode_indo.append(additive_df.loc[additive_df['idx'] == int(id), 'anode'].values[0])
    cell_type_info.append(additive_df.loc[additive_df['idx'] == int(id), 'cell_type'].values[0])

reunify_df = pd.DataFrame()
reunify_df['idx'] = idx_each_additive
reunify_df['additives'] = additive_abbr_single
reunify_df['electrolytes'] = electrolytes_info
reunify_df['cathodes'] = cathode_info
reunify_df['anodes'] = anode_indo
reunify_df['types'] = cell_type_info
reunify_df.to_csv('./V3/processed_data/additives_order.csv', index=False)

        

