import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('./V3/processed_data/additives_all.csv')
plot_df = df[df['CEI&SEI'] != 'Unspecified']
count_df = plot_df.groupby(["cell_type", "CEI&SEI"]).size().reset_index(name="Count")

plt.figure(figsize=(6,8))
ax = sns.barplot(
    data=count_df,
    x="Count",       
    y="cell_type",     
    hue="CEI&SEI",
    orient="h"
)

plt.xlabel("Counts")
plt.ylabel("Cell Types")
plt.title("The number of CEI/SEI in different battery type")
plt.legend(title="Interface Type")
plt.tight_layout()
plt.savefig('./V3/plots/CEI_SEI_details.png', dpi=600, bbox_inches='tight')

# plt.clf()

# plt.figure(figsize=(8,6))
# ax = sns.barplot(
#     data=count_df,
#     x="Count",       
#     y="cell_type",     
#     hue="CEI&SEI",
#     orient="h"
# )

# plt.xlabel("Counts")
# plt.ylabel("Cell Types")
# plt.title("The number of CEI/SEI in different battery categories in the research of boron-containing additives")
# plt.legend(title="Interface Type")
# plt.tight_layout()
# plt.savefig('./V3/plots/CEI_SEI_details.png', dpi=600, bbox_inches='tight')