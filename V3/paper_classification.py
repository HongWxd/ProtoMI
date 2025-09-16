import os 
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm
from collections import Counter
import re
import PyPDF2

path = './V3/check_data_V2.csv'
data_df = pd.DataFrame(pd.read_csv(path))
additive_df = data_df[data_df['appropriate'] == 'Yes']
cell_systems = additive_df['battery_system'].tolist()
idx = additive_df['idx'].tolist()

cell_sys_list = []
LIB_idx = []
LMB_idx = []
SIB_idx = []
SMB_idx = []
LiS_idx = []
ZIB_idx = []
MIB_idx = []
CMB_idx = []
for index, type in zip(idx, cell_systems):
    if type.startswith('Li-ion') or 'Li-ion' in type:
        cell_sys_list.append('Li-ion')
        LIB_idx.append(index)
    elif 'Not found' in type:
        cell_sys_list.append('Other')
    elif type.startswith('Li mental') or 'Li mental' in type or 'Metallic lithium' in type or 'lithium-ion' in type or 'Lithium metal' in type or 'Li-metal' in type or 'lithium metal' in type or 'Li metal' in type or 'lithium-metal' in type or 'Lithium' in type:
        cell_sys_list.append('Li mental')
        LMB_idx.append(index)
    elif 'Sodium-ion' in type or 'sodium' in type or 'Na-based dual-ion' in type:
        cell_sys_list.append('Sodium-ion')
        SIB_idx.append(index)
    elif 'Sodium metal' in type or 'Sodium Metal' in type:
        cell_sys_list.append('Sodium metal')
        SMB_idx.append(index)
    elif 'Lithium-sulfur' in type or 'Li-S' in type or 'Li/S' in type:
        cell_sys_list.append('Li-S')
        LiS_idx.append(index)
    elif 'Mg-ion' in type or 'Magnesium' in type or 'magnesium' in type:
        cell_sys_list.append('Mg-ion')
        MIB_idx.append(index)
    elif 'supercapacitor' in type or 'Supercapacitor' in type:
        continue
    elif 'Calcium-metal' in type or 'Ca-metal' in type:
        cell_sys_list.append('Ca metal')
        CMB_idx.append(index)
    elif 'Zinc-air' in type:
        cell_sys_list.append('Zinc-air')
    elif 'zinc-ion' in type or 'Zinc-ion':
        cell_sys_list.append('Zinc-ion')
        ZIB_idx.append(index)
    else:
        type = type.split(' (')[0]
        cell_sys_list.append(type)

cell_sys_df = pd.DataFrame()
cell_sys_df['cell_sys'] = cell_sys_list

def CEI_SEI_details(papers_idx_list, system_type):
    paper_path = '/data/hwx/boron/boron_electrolyte_batteries/papers'
    for idx in tqdm(papers_idx_list, desc=f'Processing {system_type} papers'):
        details_df = pd.DataFrame(pd.read_csv(f'./V3/{system_type}_CEI_SEI_details.csv'))
        labeled_idx = details_df['idx'].tolist()

        additive = data_df.loc[data_df['idx'] == idx, 'boron_additive_abbr_name'].values[0]
        if additive == 'Not found':
            continue
        if idx in labeled_idx:
            continue
        
        path = paper_path + '/' + f'{idx}.pdf'
        try:
            with open(path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)

                full_text = ""

                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    full_text += page.extract_text()
        except:
            print(f'Paper {idx}.pdf is not found, skip')
            continue
            
        client = OpenAI(api_key="sk-846361ec44554e6dbacc9fc7a103232b", base_url="https://api.deepseek.com")

        system_prompt = """
        The user will provide some exam text. Please parse the 'question' and 'answer' and output them in JSON format. 

        EXAMPLE INPUT: 
        Suppose you are an expert in the field of batteries, here is the paper about a boron-containing electrolyte additive [additive name], please help me to find the following information:
        [paper...]
        
        Question:
        1. What kind of relationship of interaction exists between the boron-containing additive mentioned in this article and the CEI film of the battery? Form a stable and dense CEI membrane.
        2. Was there any mention in the article about the performance of the battery under high voltage after adding the additives? If so, what was the performance like? Please specify the exact voltage value. The battery with the additive has a higher capacity retention rate after 100 cycles at 4.5V, while the battery without the additive has a capacity retention rate of only 49.1%.
        3. What kind of relationship of interaction exists between the boron-containing additive mentioned in this article and the SEI film of the battery? Suppressing side reactions and reducing interface impedance.
        4. What is the optimal operating temperature for the battery that uses this additive? 20°C.

        [Note] 
        1. If the information is not found in the text, please answer 'Not found'.
        2. Please answer the question using the content in the article, and provide as detailed an answer as possible.


        EXAMPLE JSON OUTPUT:
        {
            'CEI': 'Form a stable and dense CEI membrane.',
            'High_voltage_performance_CEI': 'The battery with the additive has a higher capacity retention rate after 100 cycles at 4.5V, while the battery without the additive has a capacity retention rate of only 49.1%.',
            'SEI': 'Suppressing side reactions and reducing interface impedance.',
            'Optimal_operating_temperature': '20°C',
        }
        """

        user_prompt = f"Suppose you are an expert in the field of batteries, here is the paper about a boron-containing electrolyte additive {additive}, please help me to find the following information: {full_text}"

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={
                'type': 'json_object'
            }
        )

        try:
            labeled_answer = json.loads(response.choices[0].message.content)
        except:
            print(f'Paper {idx} JSON parse error, skip')
            continue

        answer = {
            'idx': idx,
            'additive': additive,
            'CEI': labeled_answer['CEI'],
            'High_voltage_performance_CEI': labeled_answer['High_voltage_performance_CEI'],
            'SEI': labeled_answer['SEI'],
            'Optimal_operating_temperature': labeled_answer['Optimal_operating_temperature']
        }

        add_df = pd.DataFrame([answer])
        df_new = pd.concat([details_df, add_df], axis=0, ignore_index=True)
        df_new.to_csv(f'./V3/{system_type}_CEI_SEI_details.csv', index=False)

    # save_df = pd.DataFrame()
    # save_df = pd.DataFrame(chat_answers)
    # save_df.to_csv(f'./V3/{system_type}_CEI_SEI_details.csv', index=False)
  
CEI_SEI_details(LIB_idx, 'LIB')
CEI_SEI_details(LMB_idx, 'LMB')
CEI_SEI_details(SIB_idx, 'SIB')
CEI_SEI_details(SMB_idx, 'SMB')
CEI_SEI_details(LiS_idx, 'LiS')
CEI_SEI_details(MIB_idx, 'MIB')
CEI_SEI_details(CMB_idx, 'CMB')
CEI_SEI_details(ZIB_idx, 'ZIB')
