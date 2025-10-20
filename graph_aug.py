import torch
from torch_geometric.data import Data
import random

def drop_nodes(data, drop_prob=0.2):
    """
    随机丢弃部分节点
    """
    num_nodes = data.num_nodes
    keep_mask = torch.rand(num_nodes) > drop_prob  # 保留节点的mask
    keep_indices = torch.nonzero(keep_mask, as_tuple=True)[0]
    
    # 子图抽取
    edge_index, edge_attr = subgraph(keep_indices, data.edge_index, relabel_nodes=True, edge_attr=data.edge_attr)
    
    # 保留对应的节点特征
    x = data.x[keep_indices]
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def perturb_edges(data, perturb_ratio=0.1):
    """
    随机扰动边：删除部分旧边 + 添加部分随机新边
    """
    edge_index = data.edge_index.clone()
    num_edges = edge_index.size(1)
    num_nodes = data.num_nodes

    # 删除部分边
    num_delete = int(num_edges * perturb_ratio / 2)
    mask = torch.ones(num_edges, dtype=torch.bool)
    del_indices = random.sample(range(num_edges), num_delete)
    mask[del_indices] = False
    edge_index = edge_index[:, mask]

    # 添加随机边
    num_add = num_delete
    new_edges = torch.randint(0, num_nodes, (2, num_add))
    edge_index = torch.cat([edge_index, new_edges], dim=1)

    data.edge_index = edge_index
    return data


# ========== 示例 ==========
from torch_geometric.utils import subgraph

# 构造一个简单的示例图
x = torch.randn(6, 4)  # 6个节点，每个节点4维特征
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2],
                           [1, 0, 3, 2, 5, 4, 2, 0]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

# 应用增强
data_aug1 = drop_nodes(data, drop_prob=0.3)
data_aug2 = perturb_edges(data, perturb_ratio=0.2)

print("原始节点数:", data.num_nodes)
print("增强后节点数:", data_aug1.num_nodes)
print("原始边数:", data.num_edges)
print("增强后边数:", data_aug2.num_edges)

print("原始数据:", data)
print("增强节点后:", data_aug1)
print("增强边后:", data_aug2)
