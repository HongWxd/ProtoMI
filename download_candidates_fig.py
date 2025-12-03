import requests
import pandas as pd

def download_pubchem_png(identifier, label, is_cas=False, out_name=None):
    # is_cas = True → 用 CAS 查询
    if is_cas:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{identifier}/PNG"
    else:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{identifier}/PNG"

    out_name = out_name or f"./predict_molecules/{identifier}_{label}.png"

    r = requests.get(url)
    if r.status_code == 200:
        with open(out_name, "wb") as f:
            f.write(r.content)
        print(f"Saved: {out_name}")
    else:
        print(f"Failed to download {identifier}")


# 示例：CID

filter_data = pd.read_excel('./result_files/filtered_predicted_additives.xlsx')
cids = filter_data['cid'].tolist()
labels = filter_data['label'].tolist()
for cid, label in zip(cids, labels):
    download_pubchem_png(int(cid), label)