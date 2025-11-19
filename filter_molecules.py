import requests
import pandas as pd
import json
from tqdm import tqdm

def get_pubchem_name(cid):
    """
    根据 PubChem CID 查询化合物名称（IUPAC Name 或 Common Name）
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,Title/JSON"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        props = data.get("PropertyTable", {}).get("Properties", [{}])[0]

        # 有些化合物可能缺失Title或IUPACName
        title = props.get("Title", None)
        iupac = props.get("IUPACName", None)

        return {
            "IUPACName": iupac
        }

    except Exception as e:
        print(f"Error querying CID {cid}: {e}")
        return None


# 示例
if __name__ == "__main__":
    test_cids = [2244, 5793, 1983]   # 示例：咖啡因、葡萄糖等
    molecule_path = './V3/processed_data/predicted_labels.csv'
    data_df = pd.read_csv(molecule_path)
    labels = data_df['label'].tolist()
    unique_labels = list(set(labels))

    selected_ids = {}
    for label in tqdm(unique_labels):
        if label < 4:
            continue
        print(label)

        ids = data_df.loc[data_df['label'] == label, 'id'].tolist()
        selects = []
        for cid in tqdm(ids):
            info = get_pubchem_name(cid)
            try:
                name = info['IUPACName']
            except:
                continue

            if name is None:
                continue

            if name.endswith('acid'):
                continue
            else:
                selects.append(cid)
        
        selected_ids[label] = selects

        with open(f"./V3/processed_data/filter_ids_label_{label}.json", "w", encoding="utf-8") as f:
            json.dump(selected_ids, f, ensure_ascii=False, indent=4)

