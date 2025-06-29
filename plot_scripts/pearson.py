import pickle
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

with open('./data/norm_normal.pkl', 'rb') as f:
    desp_data = pickle.load(f)

df = pd.DataFrame(desp_data)
corr_matrix = df.corr(method='pearson')
feature_name_df = pd.DataFrame(pd.read_excel('./plot_scripts/pearson_data/chemical_attributes.xlsx'))
features_name = feature_name_df['Attributes'].values.tolist()
corr_matrix.index = features_name
corr_matrix.columns = features_name

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap="Spectral")
plt.title("Feature Pearson Correlation Heatmap")
plt.tight_layout()
plt.savefig('./figs/pearson_descriptors.png', dpi=600)