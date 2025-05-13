import os 
import pandas as pd

labeled_data_df = pd.DataFrame(pd.read_csv('./data/labeled_data.csv'))
all_data_df = pd.DataFrame(pd.read_csv('./data/searching_space_data.csv'))

cids = labeled_data_df['cid'].values.tolist()
formulas = []
for cid in cids:
    formula = all_data_df.loc[all_data_df['cid'] == float(cid), 'formula'].values[0]
    formulas.append(formula)

labeled_data_df['formula'] = formulas
labeled_data_df.to_csv('./data/labeled_compounds.csv', index=False)