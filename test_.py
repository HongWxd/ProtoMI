import os 
import pandas as pd

labeled_data_df = pd.DataFrame(pd.read_csv('./data/labeled_data.csv'))
all_data_df = pd.DataFrame(pd.read_csv('./data/searching_space_data.csv'))

# weights = all_data_df['weight'].values.tolist()
# small = [i for i in weights if 100 <= i <= 200]
# print(len(small))

# weights = labeled_data_df['weight'].values.tolist()
# # small = [i for i in weights if i <= 100]
# print(max(weights), min(weights))

label = labeled_data_df['label'].values.tolist()
label_1 = [i for i in label if i == 1]
label_0 = [i for i in label if i == 0]
# small = [i for i in weights if i <= 100]
print(len(label_1), len(label_0))