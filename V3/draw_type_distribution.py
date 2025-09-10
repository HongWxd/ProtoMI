import os 
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm
from collections import Counter
import re

path = './V3/check_data_V2.csv'
data_df = pd.DataFrame(pd.read_csv(path))
additive_df = data_df[data_df['appropriate'] == 'Yes']
cell_types = additive_df['format'].tolist()
cell_types = [i for i in cell_types if i != 'Not found']
cell_systems = additive_df['battery_system'].tolist()
cell_systems = [i for i in cell_systems if i != 'Not found']

cell_type_list = []
for type in cell_types:
    if type.startswith('Coin') or 'Coin' in type:
        cell_type_list.append('Coin')
    elif 'Swagelok' in type or 'Swaglock-type' in type or 'Swagelock-type' in type:
        cell_type_list.append('Swagelok Type')
    elif 'Pellet' in type or 'pellet' in type:
        cell_type_list.append('Pellet Type')
    else:
        type = type.split(' (')[0]
        cell_type_list.append(type)

cell_type_df = pd.DataFrame()
cell_type_df['cell_type'] = cell_type_list


cell_sys_list = []
for type in cell_systems:
    if type.startswith('Li-ion') or 'Li-ion' in type:
        cell_sys_list.append('Li-ion')
    elif type.startswith('Li mental') or 'Li mental' in type or 'Metallic lithium' in type or 'lithium-ion' in type or 'Lithium metal' in type or 'Li-metal' in type or 'lithium metal' in type or 'Li metal' in type or 'lithium-metal' in type or 'Lithium' in type:
        cell_sys_list.append('Li mental')
    elif 'Sodium-ion' in type or 'sodium' in type or 'Na-based dual-ion' in type:
        cell_sys_list.append('Sodium-ion')
    elif 'Sodium metal' in type or 'Sodium Metal' in type:
        cell_sys_list.append('Sodium metal')
    elif 'Lithium-sulfur' in type or 'Li-S' in type or 'Li/S' in type:
        cell_sys_list.append('Li-S')
    elif 'Mg-ion' in type or 'Magnesium' in type or 'magnesium' in type:
        cell_sys_list.append('Mg-ion')
    elif 'supercapacitor' in type or 'Supercapacitor' in type:
        continue
    elif 'Calcium-metal' in type or 'Ca-metal' in type:
        cell_sys_list.append('Ca metal')
    elif 'Zinc-air' in type:
        cell_sys_list.append('Zinc-air')
    elif 'zinc-ion' in type or 'Zinc-ion':
        cell_sys_list.append('Zinc-ion')
    else:
        type = type.split(' (')[0]
        cell_sys_list.append(type)

cell_sys_df = pd.DataFrame()
cell_sys_df['cell_sys'] = cell_sys_list


plt.figure(figsize=(6,4))
sns.histplot(data=cell_type_df, x="cell_type", discrete=True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  
plt.savefig('./V3/cell_type_distribution.png', dpi=600)
plt.clf()


plt.figure(figsize=(6,4))
sns.histplot(data=cell_sys_df, x="cell_sys", discrete=True)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  
plt.savefig('./V3/cell_system_distribution.png', dpi=600)





