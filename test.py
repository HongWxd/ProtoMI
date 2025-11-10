import torch
import random

# ------------------- 数据 -------------------
batch_ids = [101, 102]
id2idx = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}
prototype_embeddings = torch.randn(5, 128)  # 5 prototypes
r = 2  # 负样本数量

# ------------------- 正样本 -------------------
pos_indices = [id2idx[id] for id in batch_ids]   # [0, 1]
pos_prototypes = prototype_embeddings[pos_indices]

# ------------------- 负样本 -------------------
all_proto_indices = list(range(len(prototype_embeddings)))
neg_indices = random.sample([i for i in all_proto_indices if i not in pos_indices], r)
neg_prototypes = prototype_embeddings[neg_indices]

# ------------------- 拼接 -------------------
proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)  # [pos_n + neg_n, D]

# ------------------- 生成 query -------------------
batch_size = len(batch_ids)
query = torch.randn(batch_size, 128)  # 模拟 encoder 输出

# ------------------- logits -------------------
logits_proto = torch.mm(query, proto_selected.t())  # [batch, pos_n + neg_n]

# ------------------- 正确 label -------------------
# 正样本在 proto_selected 前 pos_n 个位置
# labels_proto = torch.arange(len(pos_indices), dtype=torch.long)  # [0, 1]
labels_proto = torch.arange(len(batch_ids)).long()

print("proto_selected shape:", proto_selected.shape)
print("logits_proto shape:", logits_proto.shape)
print("labels_proto:", labels_proto)
