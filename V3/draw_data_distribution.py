import os 
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm
from collections import Counter
import re

path = './V3/DOIs.csv'
data_df = pd.DataFrame(pd.read_csv(path))

with open('./V3/papers_label.json', "r") as f:
    label_data = json.load(f)

titles = []
dois = []
cathodes = []
anodes = []
electrolytes = []
additives = []
years = []
formats = []
relevants = []
for key, value in label_data.items():
    key = key.replace('_', '/')
    try:
        title = data_df.loc[data_df['DOI'] == key, 'title'].values[0]
    except:
        title = key

    titles.append(title)
    dois.append(key)
    cathodes.append(value['cathode_material'])
    anodes.append(value['anode_material'])
    electrolytes.append(value['electrolyte'])
    additives.append(value['additive'])
    if value['year'] == 'Not found':
        years.append(0)
    
    else:
        years.append(int(value['year']))
    formats.append(value['format'])
    relevants.append(value['relevant'])

labeled_df = pd.DataFrame()
labeled_df['relevant'] = relevants
labeled_df['format'] = formats
labeled_df['cathode_material'] = cathodes
labeled_df['anode_material'] = anodes
labeled_df['electrolyte'] = electrolytes
labeled_df['additive'] = additives
labeled_df['year'] = years
labeled_df['title'] = titles
labeled_df['DOI'] = dois


remain_df = labeled_df[labeled_df['relevant'] == 'Yes']
remain_df = labeled_df[labeled_df['year'] > 0]
remain_df = remain_df[remain_df['year'] < 2025]

plt.figure(figsize=(6,4))
sns.histplot(data=remain_df, x="year", discrete=True)
plt.tight_layout()  
plt.savefig('./V3/year_distribution.png', dpi=600)





