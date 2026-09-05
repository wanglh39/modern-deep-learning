# 生成 73_gnn.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 73 — 图神经网络：消息传递与 Graph Transformer

> 🧭🌐 从欧几里得数据到图结构数据——GNN 如何在非欧空间中学习。

## 本章你将掌握

1. **图数据**：节点、边、邻接矩阵
2. **消息传递范式**：GCN、GAT、GraphSAGE
3. **Graph Transformer**：全局注意力
4. **应用**：分子图、社交网络、知识图谱""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
torch.manual_seed(42)
print("环境准备完成 ✅")""")

md("""## 1. 图数据基础

### 1.1 为什么需要 GNN?

```
传统深度学习:
  - 图像: 网格 (欧几里得)
  - 文本: 序列 (1D 欧几里得)
  - CNN/RNN 处理规则结构

现实世界很多数据是图:
  - 社交网络: 用户-用户
  - 分子: 原子-化学键
  - 知识图谱: 实体-关系
  - 引用网络: 论文-引用

图是非欧几里得的:
  - 无固定网格
  - 节点数可变
  - 无固定顺序
  → 需要 GNN
```

### 1.2 图的表示

```
图 G = (V, E):
  V: 节点集合 (vertices)
  E: 边集合 (edges)

表示方式:
  1. 邻接矩阵 A: A[i,j] = 1 if (i,j) in E
  2. 边列表: [(0,1), (1,2), ...]
  3. 邻接表: {0: [1,2], 1: [0,2], ...}

节点特征: X in R^{n x d}
  n: 节点数
  d: 特征维度
```""")

code("""# 构建一个简单的图
#    0 --- 1
#    |     |
#    2 --- 3 --- 4
edges = [(0,1), (0,2), (1,3), (2,3), (3,4)]
n_nodes = 5

# 邻接矩阵
A = np.zeros((n_nodes, n_nodes))
for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1  # 无向图

print("邻接矩阵:")
print(A)

# 节点特征 (随机)
X = np.random.randn(n_nodes, 3)
print(f"\\n节点特征 X shape: {X.shape}")

# 度矩阵
D = np.diag(A.sum(axis=1))
print(f"\\n度: {A.sum(axis=1)}")

# 可视化
fig, ax = plt.subplots(figsize=(7, 5))
pos = np.array([[0, 1], [1, 1], [0, 0], [1, 0], [2, 0]], dtype=float)
for i, j in edges:
    ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], 'b-', linewidth=2)
ax.scatter(pos[:,0], pos[:,1], s=300, c='steelblue', edgecolors='black', zorder=5)
for i in range(n_nodes):
    ax.annotate(str(i), pos[i], ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.set_title('图结构', fontsize=14, fontweight='bold')
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 1.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('notebooks/fig_graph_basic.png', bbox_inches='tight')
plt.show()""")

md("""## 2. 消息传递范式

### 2.1 统一框架

```
GNN 的核心: 消息传递 (Message Passing)

对每个节点 v:
  h_v^{l+1} = UPDATE(h_v^l, AGGREGATE({h_u^l : u in N(v)}))

  - h_v^l: 节点 v 在第 l 层的表示
  - N(v): 节点 v 的邻居
  - AGGREGATE: 聚合邻居信息 (sum/mean/max)
  - UPDATE: 更新节点表示

不同 GNN = 不同 AGGREGATE + UPDATE:
  GCN:     归一化求和
  GraphSAGE: 采样 + 聚合
  GAT:     注意力加权
```

### 2.2 GCN (图卷积网络)

```
GCN 公式:
  H^{l+1} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^l W^l)

  Ã = A + I (加自环)
  D̃: Ã 的度矩阵
  W: 可学习权重

直觉:
  - 每个节点聚合邻居特征
  - 用度归一化 (防止高度节点主导)
  - 线性变换 + 非线性
```""")

code("""# GCN 层实现
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A_hat):
        # A_hat: 归一化邻接矩阵 (D̃^{-1/2} Ã D̃^{-1/2})
        # 消息传递: AXW
        support = self.linear(X)
        out = torch.matmul(A_hat, support)
        return out

class GCN(nn.Module):
    def __init__(self, n_features, hidden, n_classes):
        super().__init__()
        self.gc1 = GCNLayer(n_features, hidden)
        self.gc2 = GCNLayer(hidden, n_classes)

    def forward(self, X, A_hat):
        H = F.relu(self.gc1(X, A_hat))
        H = self.gc2(H, A_hat)
        return F.log_softmax(H, dim=1)

# 准备数据
A_tensor = torch.tensor(A, dtype=torch.float32)
I = torch.eye(n_nodes)
A_hat = A_tensor + I
D_hat = torch.diag(A_hat.sum(dim=1))
D_hat_inv_sqrt = torch.linalg.inv(torch.sqrt(D_hat))
A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

X_tensor = torch.tensor(X, dtype=torch.float32)
labels = torch.tensor([0, 0, 1, 1, 1])  # 节点分类

# 训练
model = GCN(n_features=3, hidden=8, n_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []
for epoch in range(100):
    optimizer.zero_grad()
    out = model(X_tensor, A_norm)
    loss = F.nll_loss(out, labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"GCN 训练:")
print(f"  初始 loss: {losses[0]:.4f}")
print(f"  最终 loss: {losses[-1]:.4f}")
print(f"  预测: {out.argmax(dim=1).numpy()}")
print(f"  真实: {labels.numpy()}")""")

md("""## 3. GAT (图注意力网络)

### 3.1 注意力机制

```
GAT: 用注意力权重替代固定归一化

对节点 i 和邻居 j:
  e_ij = LeakyReLU(a^T [W h_i || W h_j])
  α_ij = softmax_j(e_ij)  # 对所有 j in N(i)
  h_i' = σ(Σ_j α_ij W h_j)

优势:
  - 自适应权重 (学习得来)
  - 可解释 (注意力可视化)
  - 不受度的影响
```""")

code("""# GAT 层实现
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.randn(2 * out_features, 1))
        self.dropout = dropout

    def forward(self, X, A):
        n = X.size(0)
        H = self.W(X)  # (n, out)

        # 计算注意力系数
        H_repeat = H.unsqueeze(1).repeat(1, n, 1)  # (n, n, out)
        H_repeat_t = H.unsqueeze(0).repeat(n, 1, 1)  # (n, n, out)
        concat = torch.cat([H_repeat, H_repeat_t], dim=2)  # (n, n, 2*out)
        e = F.leaky_relu(torch.matmul(concat, self.a).squeeze(-1))  # (n, n)

        # 只保留有边连接的
        e = e.masked_fill(A == 0, float('-inf'))
        alpha = F.softmax(e, dim=1)  # (n, n)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 消息传递
        out = torch.matmul(alpha, H)
        return out

class GAT(nn.Module):
    def __init__(self, n_features, hidden, n_classes):
        super().__init__()
        self.gat1 = GATLayer(n_features, hidden)
        self.gat2 = GATLayer(hidden, n_classes)

    def forward(self, X, A):
        H = F.elu(self.gat1(X, A))
        H = self.gat2(H, A)
        return F.log_softmax(H, dim=1)

# 训练 GAT
model_gat = GAT(n_features=3, hidden=8, n_classes=2)
optimizer = torch.optim.Adam(model_gat.parameters(), lr=0.01)

losses_gat = []
for epoch in range(100):
    optimizer.zero_grad()
    out = model_gat(X_tensor, A_tensor)
    loss = F.nll_loss(out, labels)
    loss.backward()
    optimizer.step()
    losses_gat.append(loss.item())

print(f"GAT 训练:")
print(f"  初始 loss: {losses_gat[0]:.4f}")
print(f"  最终 loss: {losses_gat[-1]:.4f}")
print(f"  预测: {out.argmax(dim=1).numpy()}")
print(f"  真实: {labels.numpy()}")

# 对比 GCN vs GAT
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses, label='GCN', linewidth=2)
ax.plot(losses_gat, label='GAT', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('GCN vs GAT 训练对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_gcn_gat.png', bbox_inches='tight')
plt.show()""")

md("""## 4. Graph Transformer

### 4.1 从局部到全局

```
GNN 的局限:
  - 只聚合局部邻居 (1-hop)
  - L 层 = L-hop 感受野
  - 远程信息传递慢

Graph Transformer:
  - 全局注意力 (所有节点互相)
  - 用位置编码保留图结构
  - 适合小-中等图

公式:
  Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
  Q = H W_Q, K = H W_K, V = H W_V

  + 拉普拉斯位置编码 (保留结构信息)
```

### 4.2 应用场景

```
GNN 应用:
  1. 节点分类: 给节点打标签
     例: 社交网络用户分类

  2. 图分类: 给整个图打标签
     例: 分子毒性预测

  3. 链接预测: 预测是否有边
     例: 推荐系统

  4. 图生成: 生成新图
     例: 药物分子设计
```""")

code("""# Graph Transformer 简化
class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, H):
        # 自注意力 (全局)
        attn_out, _ = self.attention(H, H, H)
        H = self.norm1(H + attn_out)
        # FFN
        ffn_out = self.ffn(H)
        H = self.norm2(H + ffn_out)
        return H

class GraphTransformer(nn.Module):
    def __init__(self, n_features, d_model, n_classes):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.layer = GraphTransformerLayer(d_model)
        self.classify = nn.Linear(d_model, n_classes)

    def forward(self, X):
        H = self.embed(X)
        H = self.layer(H)
        out = self.classify(H)
        return F.log_softmax(out, dim=1)

# 训练 Graph Transformer
model_gt = GraphTransformer(n_features=3, d_model=16, n_classes=2)
optimizer = torch.optim.Adam(model_gt.parameters(), lr=0.01)

losses_gt = []
for epoch in range(100):
    optimizer.zero_grad()
    out = model_gt(X_tensor)
    loss = F.nll_loss(out, labels)
    loss.backward()
    optimizer.step()
    losses_gt.append(loss.item())

print(f"Graph Transformer 训练:")
print(f"  初始 loss: {losses_gt[0]:.4f}")
print(f"  最终 loss: {losses_gt[-1]:.4f}")
print(f"  预测: {out.argmax(dim=1).numpy()}")
print(f"  真实: {labels.numpy()}")""")

md("""## 5. GNN 对比

### 5.1 三种 GNN 对比

| 模型 | 聚合方式 | 优势 | 劣势 |
|------|---------|------|------|
| GCN | 归一化求和 | 简单高效 | 固定权重 |
| GAT | 注意力加权 | 自适应、可解释 | 计算量大 |
| Graph Transformer | 全局注意力 | 远程信息 | O(n²) 复杂度 |""")

code("""# 三种 GNN 对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 对比
ax = axes[0]
ax.plot(losses, label='GCN', linewidth=2)
ax.plot(losses_gat, label='GAT', linewidth=2)
ax.plot(losses_gt, label='Graph Transformer', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('三种 GNN 训练对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)

# 特性对比
ax = axes[1]
models = ['GCN', 'GAT', 'Graph\\nTransformer']
expressive = [3, 4, 5]  # 表达力
efficiency = [5, 3, 2]  # 效率
x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, expressive, width, label='表达力', color='steelblue', alpha=0.8)
ax.bar(x + width/2, efficiency, width, label='效率', color='coral', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel('得分', fontsize=12)
ax.set_title('GNN 特性对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/fig_gnn_compare.png', bbox_inches='tight')
plt.show()

print("\\n=== GNN 总结 ===")
print("GCN: 简单高效，适合大图")
print("GAT: 自适应权重，可解释")
print("Graph Transformer: 全局信息，适合小图")""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 图数据基础 (邻接矩阵/度矩阵) | ✅ |
| 消息传递范式 (MPNN) | ✅ |
| GCN (图卷积网络) | ✅ |
| GAT (图注意力网络) | ✅ |
| Graph Transformer | ✅ |
| GNN 应用 (节点/图/链接分类) | ✅ |

### 核心 takeaway
> **GNN 通过消息传递在图结构数据上学习**——GCN 简单高效、GAT 自适应可解释、Graph Transformer 捕获全局信息。从社交网络到分子设计，GNN 是处理非欧几里得数据的利器。

### 🎉 课程完结
**恭喜！你已完成现代深度学习全部 73 个 notebook。** 从 MLP 到 Transformer，从 CNN 到 GNN，从监督学习到自监督学习，从单智能体到多智能体——你已经掌握了现代深度学习的全貌。

---

> 💬 **拓展章完成 (2/2)。课程 100% 完成 🎉**""")

output_path = "notebooks/73_gnn.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")