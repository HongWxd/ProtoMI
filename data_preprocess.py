import pickle
import json
import pandas as pd
from utils.data_loader import MoleculeDataset

# paths
labeled_path = './data/additives_year.json'
searching_space_path = './data/searching_space_data_V2.csv'

sorted_additives_path = './data/additives_year_sorted.json'
mapping_json_path = './data/year_split_mapping.json'
all_data_path = './data/all_data_year.pkl'


# =========================
# 1. Load additives with year
# =========================
with open(labeled_path, 'r', encoding='utf-8') as f:
    additives_data = json.load(f)


# =========================
# 2. Sort positive molecules by year
# =========================
positive_items = list(additives_data.items())

positive_items.sort(
    key=lambda x: (
        9999 if x[1].get("year") is None else int(x[1]["year"]),
        x[0]
    )
)

sorted_additives_data = {k: v for k, v in positive_items}

with open(sorted_additives_path, 'w', encoding='utf-8') as f:
    json.dump(sorted_additives_data, f, ensure_ascii=False, indent=2)

print(f"Saved sorted additives to: {sorted_additives_path}")


# =========================
# 3. Build mapping BEFORE preprocessing
# =========================
n_positive = len(sorted_additives_data)

year_sorted_positive_molecules = []

for idx, (name, meta) in enumerate(positive_items):
    year_sorted_positive_molecules.append(
        {
            "idx_in_all_data": idx,
            "name": name,
            "year": meta.get("year", None),
            "smiles": meta.get("smiles", None)
        }
    )

mapping = {
    "description": (
        "This mapping is generated before MoleculeDataset preprocessing. "
        "The final all_data_year.pkl should be ordered as: "
        "[year-sorted positive additives] + [unlabeled molecules]. "
        "Positive indices in this file therefore correspond to positions in all_data_year.pkl "
        "as long as MoleculeDataset keeps input order."
    ),

    "positive_data_path_original": labeled_path,
    "positive_data_path_sorted": sorted_additives_path,
    "all_data_path": all_data_path,

    "n_positive": n_positive,

    "positive_start_idx": 0,
    "positive_end_idx": n_positive - 1,

    "year_sorted_positive_molecules": year_sorted_positive_molecules,

    "cutoffs": {}
}


cutoff_years = [2017, 2019, 2021, 2023]

for cutoff in cutoff_years:
    train_indices = []
    hidden_indices = []

    for idx, (name, meta) in enumerate(positive_items):
        year = meta.get("year", None)

        if year is None:
            continue

        year = int(year)

        if year <= cutoff:
            train_indices.append(idx)
        else:
            hidden_indices.append(idx)

    last_train_positive_idx = max(train_indices) if train_indices else None
    first_hidden_positive_idx = min(hidden_indices) if hidden_indices else None

    mapping["cutoffs"][f"cutoff_{cutoff}"] = {
        "cutoff_year": cutoff,

        "train_positive_start_idx": 0,
        "train_positive_end_idx": last_train_positive_idx,

        "hidden_positive_start_idx": first_hidden_positive_idx,
        "hidden_positive_end_idx": n_positive - 1,

        "n_train_positive": len(train_indices),
        "n_hidden_positive": len(hidden_indices),

        "train_positive_range_python": (
            f"all_data[0:{last_train_positive_idx + 1}]"
            if last_train_positive_idx is not None
            else None
        ),

        "hidden_positive_range_python": (
            f"all_data[{first_hidden_positive_idx}:{n_positive}]"
            if first_hidden_positive_idx is not None
            else None
        ),

        "note": (
            f"For cutoff {cutoff}, use positive molecules with year <= {cutoff} "
            f"as USL input, and molecules with year > {cutoff} as hidden positives."
        )
    }

    print(
        f"cutoff {cutoff}: "
        f"train idx = 0-{last_train_positive_idx}, "
        f"hidden idx = {first_hidden_positive_idx}-{n_positive - 1}, "
        f"train={len(train_indices)}, hidden={len(hidden_indices)}"
    )


with open(mapping_json_path, 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f"Saved mapping BEFORE preprocessing to: {mapping_json_path}")


# # =========================
# # 4. Preprocess dataset AFTER mapping is saved
# # =========================
# searching_space_df = pd.read_csv(searching_space_path)

# dataset = MoleculeDataset(sorted_additives_data, searching_space_df)

# with open(all_data_path, 'wb') as f:
#     pickle.dump(dataset.data, f)

# print(f"Saved all data to: {all_data_path}")