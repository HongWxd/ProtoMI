import os
import pandas as pd
import numpy as np

additive_df = pd.DataFrame(pd.read_csv('./V3/processed_data/additives_all.csv'))
all_df = pd.DataFrame(pd.read_csv('./V3/processed_data/check_data_V2.csv'))
additives = additive_df['additives'].tolist()
additives_abbr = (additive_df['additives_abbr'].tolist())
idx = additive_df['idx'].tolist()

cell_types = []
cathodes = []
anodes = []
additives = []
additives_abbr = []
electrolytes = []
years = []
CEI_SEI = []
details = []
titles = []
DOIs = []
abstracts = []
for id in idx:
    cell_types.append(all_df.loc[all_df['idx'] == id, 'battery_system'].values[0])
    cathodes.append(additive_df.loc[additive_df['idx'] == id, 'cathode'].values[0])
    anodes.append(additive_df.loc[additive_df['idx'] == id, 'anode'].values[0])
    additives.append(additive_df.loc[additive_df['idx'] == id, 'additives'].values[0])
    additives_abbr.append(additive_df.loc[additive_df['idx'] == id, 'additives_abbr'].values[0])
    electrolytes.append(additive_df.loc[additive_df['idx'] == id, 'electrolyte'].values[0])
    years.append(all_df.loc[all_df['idx'] == id, 'year'].values[0])
    CEI_SEI.append(additive_df.loc[additive_df['idx'] == id, 'CEI&SEI'].values[0])
    titles.append(additive_df.loc[additive_df['idx'] == id, 'title'].values[0])
    details.append(additive_df.loc[additive_df['idx'] == id, 'details'].values[0])
    DOIs.append(additive_df.loc[additive_df['idx'] == id, 'DOI'].values[0])
    abstracts.append(additive_df.loc[additive_df['idx'] == id, 'abstract'].values[0])


reunify_df = pd.DataFrame()
reunify_df['comments'] = [None] * len(idx)
reunify_df['idx'] = idx
reunify_df['cell_types'] = cell_types
reunify_df['cathodes'] = cathodes
reunify_df['anodes'] = anodes
reunify_df['additives'] = additives
reunify_df['additives_abbr'] = additives_abbr
reunify_df['SMILES'] = [None] * len(idx)
reunify_df['electrolytes'] = electrolytes
reunify_df['years'] = years
reunify_df['CEI_SEI'] = CEI_SEI
reunify_df['titles'] = titles
reunify_df['details'] = details
reunify_df['DOIs'] = DOIs
reunify_df['abstracts'] = abstracts
reunify_df = reunify_df.sort_values(by='idx', ascending=True)
print(reunify_df)
reunify_df.to_excel('./V3/processed_data/additives_all_V2.xlsx', index=False)

        

