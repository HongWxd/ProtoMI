import json
from openai import OpenAI
import os 
import PyPDF2
from tqdm import tqdm
import pandas as pd
import re

def label_by_AI(title, text):
    client = OpenAI(api_key="sk-846361ec44554e6dbacc9fc7a103232b", base_url="https://api.deepseek.com")

    system_prompt = """
    The user will provide some exam text. Please parse the 'question' and 'answer' and output them in JSON format. 

    EXAMPLE INPUT: 
    Suppose you are an expert in the field of batteries, here is the paper called title1, please help me to find the following information:
    [text...]
    
    Question:
    1. Is the additive(s) proposed by this paper added to the battery electrolyte? Yes.
    2. What is the type of the battery electrolyte proposed by this paper? Liquid.
    3. What is the battery system tested by this paper? Li-ion battery.
    3. What is the format of the battery tested by this paper? Coin.
    4. What is the cathode material used in the battery tested by this paper? NCM811.
    5. What is the anode material used in the battery tested by this paper? Graphite.
    6. What is the electrolyte composition used in the battery tested by this paper? 1M LiPF6 in EC/EMC (3:7 by wt.) with 2 percant FEC.
    7. What is the additive proposed by this paper, find it in the paper, including full term and abbreviation of additives? Lithium Difluoro(oxalato)borate (LiDFOB).
    8. What is the year of publication of this paper? 2020.

    [Note] 
    If the information is not found in the text, please answer 'Not found'.

    EXAMPLE JSON OUTPUT:
    {
        'title': 'title1',
        'relevant': 'Yes',
        'electrolyte_type': 'Liquid',
        'battery_system': 'Li-ion battery',
        'format': "Coin',
        'cathode_material': 'NCM811',
        'anode_material": 'Graphite',
        'electrolyte': '1M LiPF6 in EC/EMC (3:7 by wt.) with 2 percant FEC',
        'additive': 'Lithium Difluoro(oxalato)borate (LiDFOB)',
        'year': '2020'
    }
    """

    user_prompt = f"Suppose you are an expert in the field of batteries, here is the paper called {title}, please help me to find the following information: {text}"

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

def check_by_AI(doi, electrolyte, additive):
    client = OpenAI(api_key="sk-846361ec44554e6dbacc9fc7a103232b", base_url="https://api.deepseek.com")

    system_prompt = """
    The user will provide some information. Please parse the 'question' and 'answer' and output them in JSON format. 

    EXAMPLE INPUT: 
    Suppose you are an expert in the field of batteries, here is the electrolyte and additive information, please help me to find the following information:
    electrolyte: [...], additive: [...]
    
    Quenstions:
    1. Whether the additives used in the electrolyte are appropriate for the battery's electrolyte requirements? Yes.
    2. If this electrolyte additive is not a reasonable one, please tell me why. [Reason...]
    3. Based on your inspection, the confirmed composition of the electrolyte is? 1M LiPF6 in EC/EMC (3:7 by wt.) with 2 percant FEC.
    4. Based on your inspection, the confirmed composition of the battery electrolyte additive is? Lithium Difluoro(oxalato)borate, Fluoroethylene carbonate.
    5. The abbreviation for the confirmed components of the battery electrolyte additive is? LiDFOB, FEC.
    6. Among the confirmed additives, those containing boron compounds are (in full name)? Lithium Difluoro(oxalato)borate.
    7. Among the confirmed additives, those containing boron compounds are (in abbreviation)? LiDFOB.


    [Note] 
    If the electrolyte additive is a reasonable one, please answer 'Reasonable' to question 2.
    If question 1 is 'No', please answer question 2, and skip questions 3, 4, 5, 6, and 7, answering them as 'Not found'.
    

    EXAMPLE JSON OUTPUT:
    {
        'appropriate': 'Yes',
        'reason': 'Reasonable',
        'electrolyte': '1M LiPF6 in EC/EMC (3:7 by wt.) with 2 percant FEC',
        'additive_full_name': 'Lithium Difluoro(oxalato)borate, Fluoroethylene carbonate',
        'additive_abbr_name': 'LiDFOB, FEC',
        'boron_additive_full_name': 'Lithium Difluoro(oxalato)borate'
        'boron_additive_abbr_name': 'LiDFOB'
    }
    """

    user_prompt = f"Suppose you are an expert in the field of batteries, here is the electrolyte and additive information, please help me to find the following information: electrolyte: {electrolyte}, additive: {additive}"

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )

    check_answer = json.loads(response.choices[0].message.content)

    return check_answer

def first_round_label():
    path = '/data/hwx/boron/boron_electrolyte_batteries/papers'
    json_path = './V3/papers_label.json'
    paper_path = os.listdir(path)
    papers = [i for i in paper_path if i.endswith('.pdf')]

    data = {}
    for i, paper in enumerate(tqdm(papers, desc='Processing papers')):
        # if i > 0:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        with open(path + '/' + paper, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
            
                full_text = ""
            
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    full_text += page.extract_text()

        paper = paper.split('.pdf')[0]
        if paper in list(data.keys()):
            print(f"{paper} already labeled, skipping...")
            continue

        labeled_answer = label_by_AI(paper, full_text)
        data[f"{labeled_answer['title']}"] = labeled_answer

        with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

def second_round_check():
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
    battery_systems = []
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
        battery_systems.append(value['battery_system'])
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
    labeled_df['battery_system'] = battery_systems
    labeled_df['DOI'] = dois


    remain_df = labeled_df[labeled_df['relevant'] == 'Yes']
    electrolyte_list = remain_df['electrolyte'].tolist()
    additives_list = remain_df['additive'].tolist()
    doi_list = remain_df['DOI'].tolist()
    format_list = remain_df['format'].tolist()
    cathode_list = remain_df['cathode_material'].tolist()
    anode_list = remain_df['anode_material'].tolist()
    year_list = remain_df['year'].tolist()
    title_list = remain_df['title'].tolist()
    battery_system_list = remain_df['battery_system'].tolist()

    appropriate_list = []
    reason_list = []
    electrolyte_confirmed_list = []
    additive_full_name_list = []
    additive_abbr_name_list = []
    boron_additive_full_name_list = []
    boron_additive_abbr_name_list = []
    check_doi_list = []
    idx_list = []
    abstract_list = []
    abs_df = pd.DataFrame(pd.read_csv('./V3/boron electrolyte batteries.csv'))
    for idx, (doi, electrolyte, additive) in enumerate(zip(tqdm(doi_list), electrolyte_list, additives_list)):
        check_answer = check_by_AI(doi, electrolyte, additive)
        abs = abs_df.loc[abs_df['DOI'] == doi, 'Abstract'].values
        idx_list.append(idx + 1)
        check_doi_list.append(doi)
        abstract_list.append(abs)
        appropriate_list.append(check_answer['appropriate'])
        reason_list.append(check_answer['reason'])
        electrolyte_confirmed_list.append(check_answer['electrolyte'])
        additive_full_name_list.append(check_answer['additive_full_name'])
        additive_abbr_name_list.append(check_answer['additive_abbr_name'])
        boron_additive_full_name_list.append(check_answer['boron_additive_full_name'])
        boron_additive_abbr_name_list.append(check_answer['boron_additive_abbr_name'])


    check_df = pd.DataFrame()
    check_df['idx'] = idx_list
    check_df['DOI'] = check_doi_list
    check_df['title'] = title_list
    check_df['appropriate'] = appropriate_list
    check_df['reason'] = reason_list
    check_df['format'] = format_list
    check_df['battery_system'] = battery_system_list
    check_df['cathode_material'] = cathode_list
    check_df['anode_material'] = anode_list
    check_df['electrolyte_confirmed'] = electrolyte_confirmed_list
    check_df['additive_full_name'] = additive_full_name_list
    check_df['additive_abbr_name'] = additive_abbr_name_list
    check_df['boron_additive_full_name'] = boron_additive_full_name_list
    check_df['boron_additive_abbr_name'] = boron_additive_abbr_name_list
    check_df['abstract'] = abstract_list
    check_df['year'] = year_list
    print(check_df)
    check_df.to_csv('./V3/check_data_V2.csv', index=False)


first_round_label()
# second_round_check()

# V2_df = pd.DataFrame(pd.read_csv('./V3/check_data_V2.csv'))
# boron_additive_name = V2_df['boron_additive_full_name'].to_list()
# boron_additive_name = list(set(boron_additive_name))
# for name in boron_additive_name:
#     additives = re.split(r'(?<=[A-Za-z])\s*,\s*(?=[A-Za-z])', name)
#     print((additives))
    # print("PubChem:", PUBCHEM(name))
    # print("OPSIN:", OPSIN(name))

# print(boron_additive_name)