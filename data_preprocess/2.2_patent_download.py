from selenium import webdriver
from selenium.webdriver.common.by import By
import os
from pywinauto.application import Application
from tqdm import tqdm
import time
import pyautogui
import pandas as pd

def download_one_patent(patent_id,file_name):
    driver = webdriver.Edge()

    # login
    driver.get("https://patents.google.com/")
    time.sleep(15)

    input_box = driver.find_element(By.XPATH, "/html/body/search-app/landing-page/div/search-box/div/div/div[1]/input").send_keys(patent_id)
    time.sleep(2)

    button1 = driver.find_element(By.XPATH,
                                  "/html/body/search-app/landing-page/div/search-box/div/button").click()
    time.sleep(5)

    try:
        download_button = driver.find_element(By.XPATH,'/html/body/search-app/search-result/search-ui/div/div/div/div/div/result-container/patent-result/div/div/div/div[1]/div[2]/section/header/div/a').click()
        time.sleep(10)

        pyautogui.keyDown('ctrl')
        pyautogui.press('s')
        pyautogui.keyUp('ctrl')
        time.sleep(10)

        pyautogui.typewrite(f'{file_name}')

        # find the window
        app = Application().connect(title_re="另存为")

        # new a folder
        new_folder_button_location = pyautogui.locateCenterOnScreen('patents.png')
        if new_folder_button_location:
            time.sleep(2)

            pyautogui.moveTo(new_folder_button_location)
            pyautogui.doubleClick()
            time.sleep(5)

        # click the save button to save the PDF file
        save_button_location = pyautogui.locateCenterOnScreen('save_button.png')
        if save_button_location:
            time.sleep(2)

            pyautogui.moveTo(save_button_location)
            pyautogui.click()
            time.sleep(8)

        driver.quit()

    except:
        with open('./skip_list.txt', 'a+') as f:
            f.write(file_name+'\n')
        driver.quit()

path = './patents/'
df = pd.DataFrame(pd.read_csv('./processed_data/label_data.csv'))
pid_df = df['pid'].values.tolist()
doi_df = df['doi'].values.tolist()
dois = []
pids = []
manual_paper = []
for pid, doi in zip(pid_df, doi_df):
    if str(doi) == 'nan':# need to manually download
        manual_paper.append(df.loc[df['pid'] == pid, 'literatures'].values)
    elif str(pid) == str(doi):# they are patents
        pids.append(doi)
    else:
        dois.append(doi)# they are published papers
pids = set(pids)

for patent_id in tqdm(pids, desc='Downloading patents: '):
    path = 'C:/Users/10704/Downloads/patents/'
    if not os.path.exists(path):
        os.makedirs(path)

    files_path = os.listdir(path)
    files = [i.split('.pdf')[0] for i in files_path if i.endswith('.pdf') or i.endswith('.PDF')]
    file_name = patent_id.replace('-', '')
    skip_list = []
    with open('./skip_list.txt', 'r') as file:
        for line in file:
            skip_list.append(line.strip())

    if file_name in files or file_name in skip_list:
        continue

    download_one_patent(patent_id, file_name)
