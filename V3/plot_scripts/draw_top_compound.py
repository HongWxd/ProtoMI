import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

additive_df = pd.DataFrame(pd.read_csv('./V3/processed_data/additives_order.csv'))
additives = additive_df['additives'].tolist()

top_num = 21
counter = Counter(additives)

new_counter = {}
for k, v in counter.items():
    if 'BN' in k:
        continue
    else:
        new_counter[k] = v


counter = Counter(new_counter)
top = counter.most_common(top_num)
labels, values = zip(*top)

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values)
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        bar.get_height(),                  
        str(value),                        
        ha='center', va='bottom'           
    )
plt.xticks(rotation=45, ha='right')
plt.xlabel("Additives Name")
plt.ylabel("Reported Time")
plt.title(f"The top {top_num} most frequently occurring elements")
plt.tight_layout()
plt.savefig('./V3/plots/additive_count.png', dpi=600)