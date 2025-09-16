import os 
from tqdm import tqdm
import pandas as pd
import numpy as np

check_df = pd.DataFrame(pd.read_csv('./V3/check_data_V2.csv'))
dois = check_df['DOI'].tolist()
idx = check_df['idx'].tolist()

paper_path = '/data/hwx/boron/boron_electrolyte_batteries/papers/'
files_path = os.listdir(paper_path)
papers = [i for i in files_path if i.endswith('.pdf')]

for paper in tqdm(papers):
    paper_name = paper.split('.pdf')[0]
    try:
        paper_new_name = paper_name.replace('_', '/')
        idx = check_df.loc[check_df['DOI'] == paper_new_name, 'idx'].values[0]
        new_name = str(idx) + '.pdf'
        os.rename(paper_path + paper, paper_path + new_name)
    except:
        continue

