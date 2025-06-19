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

xgb_auc = XGBoost_data['AUC'].values.tolist()
gp_auc = GP_data['AUC'].values.tolist()
dt_auc = DT_data['AUC'].values.tolist()
rf_auc = RF_data['AUC'].values.tolist()
svm_auc = SVM_data['AUC'].values.tolist()
gcn_auc = GCN_data['AUC'].values.tolist()
gine_auc = GINE_data['AUC'].values.tolist()

auc_all = gp_auc + dt_auc + rf_auc + svm_auc + gcn_auc + gine_auc
model_all = (
    # ["XGBoost"] * len(xgb_auc) +
    ["GP"] * len(gp_auc) +
    ["DT"] * len(dt_auc) +
    ["RF"] * len(rf_auc) +
    ['SVM'] * len(svm_auc) + 
    ["GCN"] * len(gcn_auc) + 
    ["GINE"] * len(gine_auc)
)

df = pd.DataFrame({
    "Model": model_all,
    "AUC": auc_all
})

plt.figure(figsize=(8, 5))
sns.violinplot(x="Model", y="AUC", data=df, palette=sns.color_palette("Spectral"))
plt.title("AUC Score in Different Model after 10-folds Cross Validation")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig('./figs/violin.png', dpi=600)
