import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import numpy as np


# 输入DOI列表
battery_AI_df = pd.read_csv('./boron_electrolyte_batteries_AI.csv')
dois = battery_AI_df['DOI'].tolist()

def download_paper(doi, path):
    head = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36"
    }

    url = "https://www.sci-hub.ren/" + doi + "#"
    
    try:
        download_url = ""
        
        r = requests.get(url, headers=head)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        if soup.iframe == None:
            download_url = soup.embed.attrs["src"]
        else:
            download_url = soup.iframe.attrs["src"]
        
        # download the papers
        # print('Paper:', doi + " is downloading.")
        download_r = requests.get(download_url, headers=head)
        download_r.raise_for_status()
        with open(path + doi.replace("/", "_") + ".pdf", "wb+") as temp:
            temp.write(download_r.content)
        # print(doi + " paper has been downloaded!")

    # record the error messages
    except Exception as e:
        with open(path + "error.log", "a+") as error:
            error.write(doi + "\tfail to download the paper.\n")
            print(doi, 'fail to download!')
            # if download_url.startswith("https://"):
            #     error.write("Downloading url is: " + download_url + "\n")

for doi in dois:
    if pd.isna(doi):
        continue
    
    path = './papers/'
    # Create a folder to save the papers
    if not os.path.exists(path):
        os.mkdir(path)
    
    # Download the paper
    download_paper(doi, path)