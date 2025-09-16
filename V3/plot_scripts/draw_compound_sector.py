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
data = dict(counter)

sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))

labels = list(sorted_data.keys())
sizes = list(sorted_data.values())

top_num = 8
top_labels = labels[:top_num]
top_sizes = sizes[:top_num]

# 自定义百分比显示函数
def autopct_func(pct, allvals, index):
    absolute = int(round(pct/100.*sum(allvals)))
    if index < top_num:  # 只显示前15
        return f"{pct:.1f}%"
    else:
        return ''  # others 不显示百分比

# 绘图
plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,  # 初始都给标签
    autopct=lambda pct: None,  # 占位，稍后手动处理
    startangle=140,
    textprops={'fontsize': 8}
)

# 处理百分比文本
for i, a in enumerate(autotexts):
    a.set_text(autopct_func(wedges[i].theta2 - wedges[i].theta1, sizes, i))

# 控制标签显示（避免重叠）：前15全显示，其余每隔10个显示一次
for i, txt in enumerate(texts):
    if i < top_num:
        txt.set_visible(True)
    else:
        if i % 10 == 0:  # 每隔10个显示一次
            txt.set_visible(True)
        else:
            txt.set_visible(False)

plt.title("Pie Graph for Boron Additives (Top 15 with %)")
plt.tight_layout()
plt.savefig('./V3/sector_compound.png', dpi=600)
