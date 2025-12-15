import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

predict_molecules = pd.read_csv('./result_files/predicted_labels.csv')
locations_df = pd.read_csv('./result_files/umap_locations_all_proto.csv')

predict_mol_ids = predict_molecules['id'].tolist()
predict_locations_df = locations_df[locations_df['ID'].isin(predict_mol_ids)]

predict_locations_id = predict_locations_df['ID'].tolist()
prototypes = []
for id in predict_locations_id:
    prototype = predict_molecules.loc[predict_molecules['id'] == id, 'label'].values[0]
    prototypes.append('Prototype ' + str(prototype))

predict_locations_df['prototypes'] = prototypes


background_df_id = locations_df[~locations_df['ID'].isin(predict_mol_ids)]['ID'].tolist()
positive_ids = [i for i in range(1, 127)]
positive_ids += [int(str(i) + '000') for i in range(1, 127)]
positive_ids += [int(str(i) + '001') for i in range(1, 127)]
positive_ids += [int(str(i) + '010') for i in range(1, 127)]
positive_ids += [int(str(i) + '011') for i in range(1, 127)]
positive_ids += [int(str(i) + '100') for i in range(1, 127)]
ids = []
for id in background_df_id:
    if id in positive_ids:
        continue

    ids.append(id)

background_locations_df = locations_df[locations_df['ID'].isin(ids)]
positive_locations_df = locations_df[locations_df['ID'].isin(positive_ids)]
positive_locations_df.to_csv('./result_files/positive_samples_location.csv', index=False)
background_label = ['Not recommended'] * len(background_locations_df)
background_locations_df['prototypes'] = background_label

predict_locations_df = pd.concat([background_locations_df, predict_locations_df], axis=0)

order = [
    "Not recommended",
    "Prototype 1",
    "Prototype 2",
    "Prototype 3",
    "Prototype 4",
    "Prototype 5",
    "Prototype 6",
    "Prototype 7",
]
predict_locations_df["prototypes"] = pd.Categorical(
    predict_locations_df["prototypes"],
    categories=order,
    ordered=True
)

predict_locations_df = predict_locations_df.sort_values("prototypes").reset_index(drop=True)
predict_locations_df = predict_locations_df[~predict_locations_df["ID"].isin([-1, -2, -3, -4, -5, -6, -7])]
predict_locations_df.to_csv('./result_files/predicted_locations.csv', index=False)

# umap_df = pd.DataFrame()
# UMAP1 = predict_locations_df['UMAP1']
# UMAP2 = predict_locations_df['UMAP2']
# umap_df['UMAP1'] = UMAP1
# umap_df['UMAP2'] = UMAP2
# umap_df['Prototype'] = predict_locations_df['prototypes'].values

# plt.figure(figsize=(10, 8))
# sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Prototype', palette='tab10', s=20, edgecolor=None, alpha=0.7)

# plt.title('UMAP of Graph Embeddings by Prototype', fontsize=16)
# plt.xlabel('UMAP 1', fontsize=14)
# plt.ylabel('UMAP 2', fontsize=14)

# plt.legend(title='Prototype', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('./result_files/predict_result_all.png', dpi=1000)


