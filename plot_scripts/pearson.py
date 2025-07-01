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
print(len(features_name))
corr_matrix.index = features_name
corr_matrix.columns = features_name

abs_corr = corr_matrix.abs()
np.fill_diagonal(abs_corr.values, 0)

mask = (abs_corr <= 0.8).all(axis=1)
retained_features = abs_corr.index[mask].tolist()

print(f"保留的特征数量: {len(retained_features)}")
print(f"保留的特征列表:\n{retained_features}")

filtered_corr = corr_matrix.loc[retained_features, retained_features]
print(filtered_corr)

plt.figure(figsize=(12, 10))
sns.heatmap(filtered_corr, annot=False, cmap="Spectral")
plt.title("Feature Pearson Correlation Heatmap")
plt.tight_layout()
plt.savefig('./figs/pearson_descriptors.png', dpi=600)