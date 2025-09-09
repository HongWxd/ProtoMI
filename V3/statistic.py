import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

data_df = pd.DataFrame(pd.read_csv('./V3/check_data_V2.csv'))
boron_additive = data_df['boron_additive_abbr_name'].tolist()
total_list = []
for additives in boron_additive:
    if additives == 'Not found' or additives == 'Not specified in input, but examples are (C6H3F)O2B(C6H3F2), (C6F4)O2B(C6F5)':
        continue
    
    additive_list = additives.split(',')
    if len(additive_list) > 1:
        total_list += additive_list
    else:
        total_list.append(additives)

top_num = 15
counter = Counter(total_list)
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
plt.savefig('./V3/additive_count.png', dpi=600)

print(len(total_list))
print(len(set(total_list)))