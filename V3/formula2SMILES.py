import requests
import urllib.parse
import requests
import urllib.parse

def OPSIN(name):
    # URL 编码 (避免括号、空格等导致错误)
    encoded_name = urllib.parse.quote(name)
    url = f"https://opsin.ch.cam.ac.uk/opsin/{encoded_name}.smi"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    else:
        return None



def PUBCHEM(name):
    encoded = urllib.parse.quote(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/JSON"
    r = requests.get(url)
    if r.status_code == 200:
        try:
            return r.json()['PropertyTable']['Properties'][0]['ConnectivitySMILES']
        except KeyError:
            return None
    else:
        return None


API_KEY = "Ry0kgf5EnI72PBnurxrmj8vS7Dus3a3f6bWr5do4"

def chemspider(name):
    url = "https://api.rsc.org/compounds/v1/filter/name"
    headers = {"apikey": API_KEY}
    payload = {"name": name}
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        results = r.json().get("results", [])
        if results:
            csid = results[0]
            # 再用csid查询SMILES
            url2 = f"https://api.rsc.org/compounds/v1/records/{csid}/smiles"
            r2 = requests.get(url2, headers=headers)
            if r2.status_code == 200:
                return r2.text.strip('"')
    return None



# 示例
name = "triol borate"
print("PubChem:", PUBCHEM(name))
print("OPSIN:", OPSIN(name))
print("chemspider:", chemspider(name))
