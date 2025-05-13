import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import pandas as pd
import threading
import time

# paper downloading function
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

# Download the papers by their doi number
def download_by_doi(path, dois):
    # path = './papers/'
    # Create a folder to save the papers
    if not os.path.exists(path):
        os.mkdir(path)
    files = [i.split('.pdf')[0].replace('_', '/') for i in os.listdir(path) if i.endswith('.pdf')]
    error_files = []
    with open('./papers/error.log', 'r') as file:
        for line in file:
            doi = str(line.split('	fail to download the paper.')[0])
            error_files.append(doi)

    # read the doi.txt file to prepare the downloading
    # multithread downloading
    threads = []
    for doi in tqdm(dois, desc='Downloading published papers: '):
        if str(doi) in files or str(doi) in error_files:
            continue

        print('downloading: ', doi)  
        download_paper(doi, path)

    #     t = threading.Thread(target=download_paper, args=(doi,path,))
    #     threads.append(t)
    
    # for t in threads:
    #     t.start()

    # for t in threads:
    #     t.join()

# path = '/data/hwx/boron/papers/'
path = '/data/hwx/boron/papers/'
df = pd.DataFrame(pd.read_csv('./PubChem/processed_data/label_data.csv'))
pid_df = df['pid'].values.tolist()
doi_df = df['doi'].values.tolist()
dois = []
pids = []
manual_paper = []
manual_dois = []
for pid, doi in zip(pid_df, doi_df):
    if str(doi) == 'nan':# need to manually download, these papers don't have doi number
        manual_paper.append(df.loc[df['pid'] == pid, 'literatures'].values[0])
        manual_dois.append('')
    elif str(pid) == str(doi):# they are patents
        pids.append(doi)
    else:
        dois.append(doi)# they are published papers
# dois = set(dois)
print(len(dois))
# print(len(manual_dois))

# save the doi number in a text file
with open(path + 'doi.txt', 'w') as f:
    for item in dois:
        f.write(f"{item}\n")

# download the papers
# download_by_doi(path, dois)

# check the number of downloaded papers
files = [i for i in os.listdir(path) if i.endswith('.pdf')]
print('papers number:', len(files))

# # read error.log file to gather all papers needed to download manually
# with open('./papers/error.log', 'r') as file:
#     for line in file:
#         doi = str(line.split('	fail to download the paper.')[0])
#         # print(df.loc[df['doi'] == doi, 'literatures'].values[0])
#         manual_paper.append(df.loc[df['doi'] == doi, 'literatures'].values[0])
#         manual_dois.append(doi)

# print('papers need to be downloaded manually: ', len((manual_paper)))

# manual_df = pd.DataFrame()
# manual_df['doi'] = manual_dois
# manual_df['title'] = manual_paper
# manual_df.to_csv('./manual_papers.csv', index=False)