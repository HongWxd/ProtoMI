import os 
import csv
from openai import OpenAI
import PyPDF2
import pandas as pd
from tqdm import tqdm

def label_paper_by_deepseek(label_data_df):
    """label the papers data by DeepSeek.

    Args: 
        label_data_df: the data of compounds which are needed to be labeled.

    Returns:
        None.

    Raises:
        the labeled file will be saved under the [saving_path].

    """

    client = OpenAI(api_key="sk-846361ec44554e6dbacc9fc7a103232b", base_url="https://api.deepseek.com")

    path = './papers/'
    papers_path = os.listdir(path)
    papers = [i for i in papers_path if i.endswith('.pdf')]
    label_results_df = pd.DataFrame()
    labels = []
    battery_types = []
    literature_ids = []
    formulas = []
    cids = []
    SMILEs = []
    count = 0
    for paper in tqdm(papers, desc='Labeling...'):
        print(paper)

        final_papers_label_df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/final_papers_label.csv'))
        labeled_papers_list = final_papers_label_df['literature_id'].values.tolist()
        labeled_list = set([i for i in labeled_papers_list])
        literature_id = paper.split('.pdf')[0].replace('_', '/')
        pdf_abs_path = path + paper
        if literature_id in labeled_list or literature_id == '10.1016/j.aca.2019.05.041' or literature_id == '10.1002/(sici)1522-2683(20000201)21:3<563::aid-elps563>3.0.co;2-5':
            continue

        with open(pdf_abs_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
        
            full_text = ""
        
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                full_text += page.extract_text()
        
        doi = label_data_df.loc[label_data_df['doi'] == literature_id, 'doi'].values
        if len(doi) == 0:
            smiles = label_data_df.loc[label_data_df['literatures'] == literature_id, 'SMILES'].values
        else:
            smiles = label_data_df.loc[label_data_df['doi'] == literature_id, 'SMILES'].values
        
        # if one paper related to many compounds
        for smile in smiles:
            print(smile, literature_id)
            cid_df = label_data_df[label_data_df['SMILES'] == smile]
            try:
                cid = cid_df.loc[cid_df['doi'] == literature_id, 'cid'].values[0]
            except:
                cid = cid_df.loc[cid_df['SMILES'] == smile, 'cid'].values[0]
            compound = cid_df.loc[cid_df['SMILES'] == smile, 'formula'].values[0]
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "假设你是有机电化学领域的专家，"},
                    {"role": "user", "content": f"请阅读一下文章，并基于文章的内容按照我的要求回答: {full_text}"},
                    {"role": "user", "content": "按照以下逻辑帮我提取文章中的信息："
                    "1. 根据标题帮我判断这篇文章是否是电池领域内的研究工作（包括研究电解液，电池循环寿命，电极，添加剂，电池材料，电池界面），如果是，进行下一步，如果不是则直接跳转到最后一步输出搜索结果，"
                    "2. 找到所有描述化合物公式为{compound}的全部内容，"
                    "3. 请帮我查看一下你找到的全部内容中是否有以下两个内容：(1)这个化合物对于电池反应过程中形成更好的SEI膜是否有促进作用，(2)这个化合物被用于什么电池，"
                    "4. 严格按照以下格式回复（被[]圈起来的是变量，需要你从pdf中提取，如果提取不到的留空处理，请勿回复其他内容）：、\n是否对形成更好的SEI膜有促进作用:[如果有，返回1；如果没有，返回0；如果文章不是研究电池领域的工作，返回-1];[如果是锂离子电池返回LIB，钠离子电池返回SIB，锌离子电池返回ZIB，镁离子电池返回MIB，锂金属电池返回LMB,剩余其他电池返回Other]"}
                ],
                stream=False
            )
            response_content = response.choices[0].message.content
            label = response_content.split(';')[0].split(':')[1].split('[')[1].split(']')[0]
            battery_type = response_content.split(';')[1]
            if battery_type == '[]':
                battery_type = -1
            else:
                battery_type = battery_type.split('[')[1].split(']')[0]
            
            new_data = [cid, literature_id, smile, compound, battery_type, label]
            saving_path = './PubChem/processed_data/final_papers_label.csv'
            with open(saving_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(new_data)
            
            print(smile, literature_id, battery_type, label)

    #         labels.append(label)
    #         battery_types.append(battery_type)
    #         literature_ids.append(literature_id)
    #         formulas.append(compound)
    #         cids.append(cid)
    #         SMILEs.append(smile)
        
    # label_results_df['cid'] = cids
    # label_results_df['literature_id'] = literature_ids
    # label_results_df['SMILE'] = SMILEs
    # label_results_df['formula'] = formulas
    # label_results_df['type'] = battery_types
    # label_results_df['label'] = labels
    # label_results_df.to_csv('./PubChem/processed_data/final_papers_label.csv', index=False)

def label_patent_by_deepseek(label_data_df):
    """label the patents data by DeepSeek.

    Args: 
        label_data_df: the data of compounds which are needed to be labeled.

    Returns:
        None.

    Raises:
        the labeled file will be saved under the [saving_path].

    """


    client = OpenAI(api_key="sk-846361ec44554e6dbacc9fc7a103232b", base_url="https://api.deepseek.com")

    path = '/data/hwx/boron/patents/'
    papers_path = os.listdir(path)
    papers = [i for i in papers_path if i.endswith('.pdf')]
    label_results_df = pd.DataFrame()
    labels = []
    battery_types = []
    literature_ids = []
    formulas = []
    cids = []
    SMILEs = []
    error_smile = []
    error_literature = []
    count = 0
    for paper in tqdm(papers, desc='Labeling...'):
        print(paper)

        final_patents_label_df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/final_patents_label.csv'))
        labeled_papers_list = final_patents_label_df['literature_id'].values.tolist()
        labeled_list = set([i for i in labeled_papers_list])
        literature_id = str(paper).split('.pdf')[0]
        pdf_abs_path = path + paper
        if literature_id in labeled_list or literature_id.startswith('JP2017017033A') or literature_id.startswith('JP2016164877A'):
            continue

        with open(pdf_abs_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
        
            full_text = ""
        
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                full_text += page.extract_text()
        
        doi_numbers = label_data_df['doi'].values
        doi_map = {}
        for doi_number in doi_numbers:
            new_doi = str(doi_number).replace('-','')
            doi_map[new_doi] = str(doi_number)
        
        doi_id = doi_map[literature_id]
        smiles = label_data_df.loc[label_data_df['doi'] == doi_id, 'SMILES'].values

        # if one paper related to many compounds
        for smile in smiles:
            print(smile, literature_id)
            cid_df = label_data_df[label_data_df['SMILES'] == smile]
            cid = cid_df.loc[cid_df['doi'] == doi_id, 'cid'].values[0]
            compound = cid_df.loc[cid_df['SMILES'] == smile, 'formula'].values[0]
            
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "假设你是有机电化学领域的专家，"},
                        {"role": "user", "content": f"请阅读一下文章，并基于文章的内容按照我的要求回答: {full_text}"},
                        {"role": "user", "content": "按照以下逻辑帮我提取文章中的信息："
                        "1. 根据标题帮我判断这篇文章是否是电池领域内的研究工作（包括研究电解液，电池循环寿命，电极，添加剂，电池材料，电池界面），如果是，进行下一步，如果不是则直接跳转到最后一步输出搜索结果，"
                        "2. 找到所有描述化合物公式为{compound}的全部内容，"
                        "3. 请帮我查看一下你找到的全部内容中是否有以下两个内容：(1)这个化合物对于电池反应过程中形成更好的SEI膜是否有促进作用，(2)这个化合物被用于什么电池，"
                        "4. 严格按照以下格式回复（被[]圈起来的是变量，需要你从pdf中提取，如果提取不到的留空处理，请勿回复其他内容）："
                        "\n是否对形成更好的SEI膜有促进作用:[如果有，返回1；如果没有，返回0；如果文章不是研究电池领域的工作，返回-1];"
                        "[如果是锂离子电池返回LIB，钠离子电池返回SIB，锌离子电池返回ZIB，镁离子电池返回MIB，锂金属电池返回LMB,剩余其他电池返回Other]"}
                    ],
                    stream=False
                )
                response_content = response.choices[0].message.content
                label = response_content.split(';')[0].split(':')[1].split('[')[1].split(']')[0]
                battery_type = response_content.split(';')[1]
                if battery_type == '[]':
                    battery_type = -1
                else:
                    battery_type = battery_type.split('[')[1].split(']')[0]
                
                new_data = [cid, literature_id, smile, compound, battery_type, label]
                saving_path = './PubChem/processed_data/final_patents_label.csv'
                with open(saving_path, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(new_data)
                
                print(smile, literature_id, battery_type, label)
            except:# for the token limitation, some patents can not be labeled, we drop these patents and record in the error.csv file
                print(smile, literature_id, 'errors')
                error_smile.append(smile)
                error_literature.append(literature_id)
                
    error_df = pd.DataFrame()
    error_df['smile'] = error_smile
    error_df['literature'] = error_literature
    error_df.to_csv('./PubChem/processed_data/error.csv', index=False)


    #         labels.append(label)
    #         battery_types.append(battery_type)
    #         literature_ids.append(literature_id)
    #         formulas.append(compound)
    #         cids.append(cid)
    #         SMILEs.append(smile)
        
    # label_results_df['cid'] = cids
    # label_results_df['literature_id'] = literature_ids
    # label_results_df['SMILE'] = SMILEs
    # label_results_df['formula'] = formulas
    # label_results_df['type'] = battery_types
    # label_results_df['label'] = labels
    # label_results_df.to_csv('./PubChem/processed_data/final_patents_label.csv', index=False)


label_data_df = pd.read_csv('./PubChem/processed_data/label_data_stage4.csv')
# label_paper_by_deepseek(label_data_df)
label_patent_by_deepseek(label_data_df)