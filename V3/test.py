import os 
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm
from collections import Counter
import re


def summarize_by_AI(text):
    client = OpenAI(api_key="sk-846361ec44554e6dbacc9fc7a103232b", base_url="https://api.deepseek.com")

    system_prompt = """
    The user will provide some string list. Please parse the 'question' and 'answer' and output them in JSON format. 

    EXAMPLE INPUT: 
    Suppose you are an expert in the field of batteries, here is the list, please help me to summarize it:
    ['TMSB', 'Boron Nitride (BN)', 'Boron nitride (BN) coating', 'Tris(pentafluorophenyl)borane (TPFPB)', 'TPFPB (tris(pentafluorophenyl)borane)', 'hBN (hexagonal boron nitride)', 'Trimethyl borate (TMB)', 'BN (Boron Nitride)', 'Boron-containing cross-linkers (LBC and TBC)', 'Ammonia borane (AB)', 'B[C2HBNO(CN)2]3, B[C2HBNS(CN)2]3, B[C4H3BN(CN)2]3', 'B2O3', 'BNNS', 'Boron nitride (BN)', 'Porous boron-containing covalent organic frameworks (COFs)', 'LiDFOB', 'B-PEG (poly(ethylene glycol)-borate ester)', 'Not found', 'Boron (B-doping in cathode, not electrolyte additive)', 'BN nanosheets (BNNS)']
    
    Please help me count the frequency of each substance in this list. Do not simply summarize based on the content that appears, but rather count according to whether there is equivalence or similarity between the entities.

    EXAMPLE JSON OUTPUT:
    {
        'Boron Nitride (BN/hBN/BNNS/coating)': 6,
        'TPFPB': 2,
        'TMSB': 1,
        'Trimethyl borate (TMB)': 1,
        'Ammonia borane (AB)': 1,
        '...': ...,
    }
    """

    user_prompt = f"Suppose you are an expert in the field of batteries, here is the list {text}, please help me to summarize it."

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )

    labeled_answer = json.loads(response.choices[0].message.content)

    return labeled_answer


path = './V3/boron_electrolyte_batteries_AI.csv'
data_df = pd.DataFrame(pd.read_csv(path))

with open('./V3/papers_label_test.json', "r") as f:
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

# labeled_df.to_csv('./V3/labeled_data.csv', index=False)

remain_df = labeled_df[labeled_df['relevant'] == 'Yes']
remain_df = labeled_df[labeled_df['year'] > 0]
remain_df = remain_df[remain_df['year'] < 2025]

# years = sorted(set(remain_df['year'].tolist()))
# most_additives = []
# for year in tqdm(years):
#     additive = remain_df[remain_df['year'] == year]['additive'].tolist()
#     additive = list(set(additive))

#     response_list = summarize_by_AI(additive)
#     max_key = max(response_list, key=response_list.get)
#     most_additives.append(max_key)

# print(years, most_additives)
# most_add_df = pd.DataFrame()
# most_add_df['year'] = years
# most_add_df['additive'] = most_additives
# most_add_df.to_csv('./V3/most_additive_per_year.csv', index=False)

plt.figure(figsize=(6,4))
sns.histplot(data=remain_df, x="year", discrete=True)
plt.tight_layout()  
plt.savefig('./V3/year_distribution.png', dpi=600)



