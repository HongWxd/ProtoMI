import os 
from tqdm import tqdm
import pandas as pd
import numpy as np

CEI_SEI_detailed_data_path = './V3/CEI_SEI_details/'
processed_df = pd.DataFrame(pd.read_csv('./V3/processed_data/check_data_V2.csv'))
files_path = os.listdir(CEI_SEI_detailed_data_path)
files = [i for i in files_path if i.endswith('.csv')]

research_field = []
content = []
supplyment = []
idx = []
additives = []
temperatures = []
dois = []
titles = []
abstract = []
cathodes = []
anodes = []
electrolytes = []
additives_abbr = []
for file in tqdm(files):
    battery_type = file.split('_CEI_SEI')[0]
    df = pd.read_csv(CEI_SEI_detailed_data_path + file)
    idx_in_df = df['idx'].tolist()

    for i in idx_in_df:
        research_field.append(battery_type)
        cei = df.loc[df['idx'] == i, 'CEI'].values[0]
        sei = df.loc[df['idx'] == i, 'SEI'].values[0]
        if cei == 'Not found' and sei != 'Not found':
            content.append('SEI')
            supplyment.append(df.loc[df['idx'] == i, 'SEI'].values[0])
        elif sei == 'Not found' and cei != 'Not found':
            content.append('CEI')
            supplyment.append(df.loc[df['idx'] == i, 'CEI'].values[0])
        else:
            content.append('Unspecified')
            supplyment.append('Unspecified')

        idx.append(i)
        additives.append(processed_df.loc[processed_df['idx'] == i, 'boron_additive_full_name'].values[0])
        additives_abbr.append(processed_df.loc[processed_df['idx'] == i, 'boron_additive_abbr_name'].values[0])
        temperatures.append(df.loc[df['idx'] == i, 'Optimal_operating_temperature'].values[0])
        dois.append(processed_df.loc[processed_df['idx'] == i, 'DOI'].values[0])
        titles.append(processed_df.loc[processed_df['idx'] == i, 'title'].values[0])
        abstract.append(processed_df.loc[processed_df['idx'] == i, 'abstract'].values[0])
        cathodes.append(processed_df.loc[processed_df['idx'] == i, 'cathode_material'].values[0])
        anodes.append(processed_df.loc[processed_df['idx'] == i, 'anode_material'].values[0])
        electrolytes.append(processed_df.loc[processed_df['idx'] == i, 'electrolyte_confirmed'].values[0])

final_df = pd.DataFrame({'idx': idx, 'cell_type': research_field, 'CEI&SEI': content,'additives': additives, 
                         'additives_abbr': additives_abbr, 'temperatures': temperatures, 'details': supplyment,
                         'DOI': dois, 'cathode': cathodes, 'anode': anodes, 'electrolyte': electrolytes, 'title': titles, 'abstract': abstract,})
final_df.to_csv('./V3/processed_data/additives_all.csv', index=False)