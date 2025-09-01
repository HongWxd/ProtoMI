import os 
import json
import pandas as pd
import numpy as np

df = pd.read_csv('./V3/boron electrolyte batteries.csv')
titles = df['Article Title'].tolist()
dois = df['DOI'].tolist()

paper_path = '/data/hwx/boron/boron_electrolyte_batteries/papers'
path = os.listdir(paper_path)
papers = [i.split('.pdf')[0] for i in path if i.endswith('.pdf')]

status = []
for doi in dois:
    if pd.isna(doi):
        status.append(0)
    else:
        doi = doi.replace('/', '_')
        if doi in papers:
            doi = doi.replace('_', '/')
            status.append(1)
        else:
            doi = doi.replace('_', '/')
            status.append('')

print(status)

new_df = pd.DataFrame()
new_df['title'] = titles
new_df['DOI'] = dois
new_df['status'] = status
new_df.to_csv('./V3/DOIs.csv', index=False)
print(new_df)

