import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

XGBoost_data = pd.DataFrame(pd.read_csv('./plot_scripts/plot_data/XGB_data.csv'))
GP_data = pd.DataFrame(pd.read_csv('./plot_scripts/plot_data/GP_data.csv'))
DT_data = pd.DataFrame(pd.read_csv('./plot_scripts/plot_data/DT_data.csv'))
RF_data = pd.DataFrame(pd.read_csv('./plot_scripts/plot_data/RF_data.csv'))
SVM_data = pd.DataFrame(pd.read_csv('./plot_scripts/plot_data/SVM_data.csv'))
GCN_data = pd.DataFrame(pd.read_csv('./plot_scripts/plot_data/GCN_data.csv'))
GINE_data = pd.DataFrame(pd.read_csv('./plot_scripts/plot_data/GINE_data.csv'))
GINE_SSL_data = pd.DataFrame(pd.read_csv('./plot_scripts/plot_data/GINE_SSL_data.csv'))
GINE_SSL_desp_data = pd.DataFrame(pd.read_csv('./plot_scripts/plot_data/GINE_SSL_descriptor_data.csv'))

xgb_auc = XGBoost_data['AUC'].values.tolist()
gp_auc = GP_data['AUC'].values.tolist()
dt_auc = DT_data['AUC'].values.tolist()
rf_auc = RF_data['AUC'].values.tolist()
svm_auc = SVM_data['AUC'].values.tolist()
gcn_auc = GCN_data['AUC'].values.tolist()
gine_auc = GINE_data['AUC'].values.tolist()
gine_ssl_auc = GINE_SSL_data['AUC'].values.tolist()
gine_ssl_d_auc = GINE_SSL_desp_data['AUC'].values.tolist()

auc_all = gp_auc + dt_auc + rf_auc + svm_auc + gcn_auc + gine_auc + gine_ssl_auc + gine_ssl_d_auc
model_all = (
    # ["XGBoost"] * len(xgb_auc) +
    ["GP"] * len(gp_auc) +
    ["DT"] * len(dt_auc) +
    ["RF"] * len(rf_auc) +
    ['SVM'] * len(svm_auc) + 
    ["GCN"] * len(gcn_auc) + 
    ["GINE"] * len(gine_auc) + 
    ["GINE_SSL"] * len(gine_ssl_auc) + 
    ["GINE_SSL_D"] * len(gine_ssl_d_auc)
)

df = pd.DataFrame({
    "Model": model_all,
    "AUC": auc_all
})

plt.figure(figsize=(8, 5))
plt.figure(figsize=(8, 5))
palette = sns.color_palette("Spectral", n_colors=8)
sns.violinplot(x="Model", y="AUC", data=df, palette=palette, inner_kws=dict(box_width=4, whis_width=2, color=".5"))
# sns.boxplot(x="Model", y="AUC", data=df, palette=palette)
sns.stripplot(x="Model", y="AUC", data=df, color='black', size=6, jitter=True, alpha=0.7)
plt.title("AUC Score in Different Model after 10-folds Cross Validation")
plt.grid(axis="y", linestyle="--", alpha=0.8)
plt.tight_layout()
plt.savefig('./figs/violin.png', dpi=600)
