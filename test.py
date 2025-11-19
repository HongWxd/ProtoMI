import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 示例数据
data = np.random.rand(10, 10)  # 10x10 矩阵

x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
x = x.flatten()
y = y.flatten()
sizes = data.flatten() * 300  # 点大小映射
colors = data.flatten()       # 颜色映射

plt.figure(figsize=(6,6))
plt.scatter(x, y, s=sizes, c=colors, cmap='magma', edgecolors='grey')
plt.gca().invert_yaxis()
plt.colorbar(label='Value')
plt.xticks(range(data.shape[1]))
plt.yticks(range(data.shape[0]))
plt.title("Bubble Heatmap Example")
plt.savefig('./test.png', dpi=300)
