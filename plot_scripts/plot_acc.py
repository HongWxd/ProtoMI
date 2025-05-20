import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os 
import seaborn as sns

df = pd.DataFrame(pd.read_excel('./plot_scripts/plot_data/test.xlsx'))
layers = df['layer'].values.tolist()
mean_acc = df['mean_acc'].values.tolist()
mean_pre = df['mean_pre'].values.tolist()
mean_re = df['mean_re'].values.tolist()
mean_f1 = df['mean_f1'].values.tolist()

plot_data = pd.DataFrame()
plot_data['layers'] = layers
plot_data['mean_acc'] = mean_acc
plot_data['mean_pre'] = mean_pre
plot_data['mean_re'] = mean_re
plot_data['mean_f1'] = mean_f1
plot_data_melted = pd.melt(plot_data, id_vars=['layers'], value_vars=['mean_acc', 'mean_pre', 'mean_re', 'mean_f1'], var_name='metric', value_name='value')

plt.figure(figsize=(10, 6))
sns.lineplot(data=plot_data_melted, x="layers", y="value", hue="metric")
for metric in plot_data_melted['metric'].unique():
    metric_data = plot_data_melted[plot_data_melted['metric'] == metric]
    plt.scatter(metric_data['layers'], metric_data['value'], label=f'{metric} 点', zorder=5)
    
plt.tight_layout()
plt.savefig('./figs/test_layers.png', dpi=600)
