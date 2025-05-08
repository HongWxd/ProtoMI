import pandas as pd
from utils.tools import Graph_data_generator


data_df = pd.DataFrame(pd.read_csv('./data/training_data.csv'))
x_smiles = data_df['smiles'].values.tolist()
y = data_df['labels'].values.tolist()
data_list = Graph_data_generator(x_smiles, y)
print(len(data_list))