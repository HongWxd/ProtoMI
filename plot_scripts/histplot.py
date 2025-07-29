import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings('ignore')

GP_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/GP_data.csv'))
GP_D_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/GP_D_True_data.csv'))

DT_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/DT_data.csv'))
DT_D_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/DT_D_True_data.csv'))

RF_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/RF_data.csv'))
RF_D_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/RF_D_True_data.csv'))

SVM_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/SVM_data.csv'))
SVM_D_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/SVM_D_True_data.csv'))

GCN_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/GCN_Dummy_D_False_SB_False_data.csv'))
GCN_D_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/GCN_Dummy_D_True_SB_False_data.csv'))
GCN_D_SB_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/GCN_Dummy_D_True_SB_True_data.csv'))

GINE_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/GINE_Dummy_D_False_SB_False_data.csv'))
GINE_D_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/GINE_Dummy_D_True_SB_False_data.csv'))
GINE_D_SB_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/GINE_Dummy_D_True_SB_True_data.csv'))

GINE_SSL_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/GINE_SSL_data.csv'))
GINE_SSL_desp_data = pd.DataFrame(pd.read_csv('./plot_scripts/violin_data/GINE_SSL_descriptor_data.csv'))

gp_auc = GP_data['AUC'].values.tolist()
gp_d_auc = GP_D_data['AUC'].values.tolist()

dt_auc = DT_data['AUC'].values.tolist()
dt_d_auc = DT_D_data['AUC'].values.tolist()

rf_auc = RF_data['AUC'].values.tolist()
rf_d_auc = RF_D_data['AUC'].values.tolist()

svm_auc = SVM_data['AUC'].values.tolist()
svm_d_auc = SVM_D_data['AUC'].values.tolist()

gcn_auc = GCN_data['AUC'].values.tolist()
gcn_d_auc = GCN_D_data['AUC'].values.tolist()
gcn_d_sb_auc = GCN_D_SB_data['AUC'].values.tolist()

gine_auc = GINE_data['AUC'].values.tolist()
gine_d_auc = GINE_D_data['AUC'].values.tolist()
gine_d_sb_auc = GINE_D_SB_data['AUC'].values.tolist()

gine_ssl_auc = GINE_SSL_data['AUC'].values.tolist()
gine_ssl_d_auc = GINE_SSL_desp_data['AUC'].values.tolist()

auc_all = [np.mean(gp_auc), np.mean(gp_d_auc), np.mean(dt_auc), np.mean(dt_d_auc), np.mean(rf_auc), np.mean(rf_d_auc), np.mean(svm_auc), np.mean(svm_d_auc), 
           np.mean(gcn_auc), np.mean(gcn_d_auc), np.mean(gine_auc), np.mean(gine_d_auc)]

model_all = (
    ["GP"] * 2 +
    ["DT"] * 2 +
    ["RF"] * 2 +
    ['SVM'] * 2 + 
    ["GCN"] * 2 + 
    ["GINE"] * 2
    # ["GCN_SSL"] * 2 +  
    # ["GINE_SSL"] * 2
)

type_all = ['without D', 'with D'] * 6

error_all = [
    np.std(gp_auc), np.std(gp_d_auc),
    np.std(dt_auc), np.std(dt_d_auc),
    np.std(rf_auc), np.std(rf_d_auc),
    np.std(svm_auc), np.std(svm_d_auc),
    np.std(gcn_auc), np.std(gcn_d_auc),
    np.std(gine_auc), np.std(gine_d_auc)
]

df = pd.DataFrame({
    "Model": model_all,
    "AUC": auc_all,
    'Class': type_all,
    'Error': error_all
})

print(df)

plt.figure(figsize=(8, 5))
palette = sns.color_palette("Spectral", n_colors=8)

ax = sns.histplot(data=df, x="Model", hue="Class", weights='AUC',
                  multiple="dodge", shrink=0.8, discrete=True)

bar_width = 0.8 / 2  # 每组两个 bar: 'with D', 'without D'
models = df["Model"].unique()
class_order = ['without D', 'with D']

for i, model in enumerate(models):
    for j, cls in enumerate(class_order):
        subset = df[(df["Model"] == model) & (df["Class"] == cls)]
        if not subset.empty:
            auc = subset["AUC"].values[0]
            err = subset["Error"].values[0]
            x = i - 0.4 + (j + 0.5) * bar_width  # 柱子的中心位置
            plt.errorbar(x=x, y=auc, yerr=err, fmt='none', ecolor='black', capsize=5, elinewidth=1)

# sns.histplot(data=df, x="Model", hue="Class", weights='AUC', multiple="dodge", shrink=.8)
plt.title("AUC Score in Different Model after 10-folds Cross Validation")
plt.grid(axis="y", linestyle="--", alpha=0.8)
plt.ylabel('AUC Score')
plt.tight_layout()
plt.savefig('./figs/histplot.png', dpi=600)
