import numpy as np
from rdkit import Chem

# 示例：一组 SMILES
smiles_list = ["B(C1=CC=C(C=C1)OC2=CC=CC=C2)(O)O", "C1=CC=CC=C1", "CC(=O)O"]

# 1. 获取所有唯一字符（字符集）
all_chars = set()
for smiles in smiles_list:
    all_chars.update(list(smiles))
all_chars = sorted(list(all_chars))  # 排序方便统一索引
char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
print("字符字典：", char_to_idx)

# 2. 编码函数
def smiles_to_onehot(smiles, max_len=50):
    onehot = np.zeros((max_len, len(char_to_idx)), dtype=np.float32)
    for i, char in enumerate(smiles):
        if i >= max_len:
            break
        onehot[i, char_to_idx[char]] = 1.0
    return onehot

# 3. 对所有 SMILES 编码
onehot_encodings = [smiles_to_onehot(smi) for smi in smiles_list]

# 示例打印
print("第一个SMILES One-Hot编码 shape:", onehot_encodings[0].shape)
print(onehot_encodings[0])
