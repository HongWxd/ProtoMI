import json
from openai import OpenAI
import os 
import PyPDF2
from tqdm import tqdm

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

path = '/data/hwx/boron/boron_electrolyte_batteries/papers'
json_path = './V3/papers_label.json'
paper_path = os.listdir(path)
papers = [i for i in paper_path if i.endswith('.pdf')]

data = {}
for i, paper in enumerate(tqdm(papers, desc='Processing papers')):
    if i > 0:
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

