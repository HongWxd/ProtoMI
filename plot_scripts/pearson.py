import pickle
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('./data/all_data_descriptors.pkl', 'rb') as f:
    all_data = pickle.load(f)

cids = []
descriptor_list = []
for data in tqdm(all_data):
    cids = data.cid
    descriptors = data.descriptors.numpy()
    descriptor_list.append(descriptors[0])
    print(descriptors[0])

df = pd.DataFrame(descriptor_list)
corr_matrix = df.corr(method='pearson')


plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Pearson Correlation Heatmap")
plt.savefig('./figs/pearson_descriptors.png', dpi=600)