# 生成 64_protein_design.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 64 — 蛋白质设计：RFdiffusion

> 🔥 从预测到设计——AI 不只能预测蛋白质结构，还能设计新蛋白质。

## 本章你将掌握

1. **蛋白质设计问题**：从功能到序列
2. **RFdiffusion**：扩散模型设计骨架
3. **ProteinMPNN**：序列设计
4. **设计验证**：AF2 验证设计""")

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

md("""## 1. 蛋白质设计问题

### 1.1 从预测到设计

```
结构预测 (AlphaFold):
  序列 → 结构 (正向问题)

蛋白质设计 (逆向问题):
  功能/结构 → 序列

  想要一个能结合X的蛋白质
  → 设计序列使其折叠成所需结构
  → 该结构有所需功能

应用:
  - 药物: 设计结合特定靶点的蛋白
  - 酶: 设计催化特定反应的酶
  - 材料: 设计有特定性质的蛋白材料
  - 疫苗: 设计新疫苗
```

### 1.2 设计流程

```
蛋白质设计流程:
  1. 确定目标功能/结构
  2. 设计骨架 (3D 结构)
     → RFdiffusion
  3. 设计序列 (氨基酸)
     → ProteinMPNN
  4. 验证 (预测结构)
     → AlphaFold2
  5. 实验验证
     → 合成并测试
```

> 💡 蛋白质设计是"逆向工程"——从想要的功能出发，设计出能实现该功能的蛋白质。""")

md("""## 2. RFdiffusion

### 2.1 核心思想

```
RFdiffusion:
  用扩散模型生成蛋白质骨架

  前向: 蛋白质结构 → 噪声
  反向: 噪声 → 蛋白质结构

  训练: 在已知蛋白质结构上训练
  生成: 从噪声开始 → 去噪 → 新结构

特点:
  - 基于 RoseTTAFold 架构
  - 条件生成: 可以指定对称性/功能位点
  - 灵活: 可以设计各种拓扑
```

### 2.2 条件设计

```
无条件: 随机生成蛋白质
有条件:
  - 指定对称性 (如 C3 对称)
  - 指定结合位点
  - 指定功能 motif
  - 指定拓扑

→ 条件让设计有目标性
```""")

code("""# RFdiffusion 简化实现
class RFdiffusionSimplified(nn.Module):
    def __init__(self, d_model=64, n_steps=50):
        super().__init__()
        self.n_steps = n_steps
        # 去噪网络
        self.net = nn.Sequential(
            nn.Linear(3 + d_model, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.feature = nn.Linear(1, d_model)

    def forward(self, n_residues, condition=None):
        # 从噪声开始
        coords = torch.randn(1, n_residues, 3) * 5

        # 条件特征
        if condition is not None:
            feat = self.feature(condition.unsqueeze(-1).unsqueeze(0))
        else:
            feat = self.feature(torch.zeros(1, n_residues, 1))

        trajectory = [coords.clone()]

        # 反向扩散
        for t in range(self.n_steps):
            inp = torch.cat([feat, coords], dim=-1)
            noise_pred = self.net(inp)
            dt = 1.0 / self.n_steps
            coords = coords - noise_pred * dt
            trajectory.append(coords.clone())

        return coords[0], trajectory

# 生成蛋白质
rf = RFdiffusionSimplified(n_steps=30)

# 无条件生成
with torch.no_grad():
    coords1, traj1 = rf(20)

    # 条件生成 (模拟对称性约束)
    condition = torch.ones(20)
    coords2, traj2 = rf(20, condition=condition)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, coords, title in [(axes[0], coords1.numpy(), '无条件生成'),
                           (axes[1], coords2.numpy(), '条件生成')]:
    ax.plot(coords[:, 0], coords[:, 1], 'b-o', markersize=4, linewidth=2)
    ax.scatter(coords[:, 0], coords[:, 1], c=range(len(coords)),
              cmap='viridis', s=50)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

plt.suptitle('RFdiffusion 蛋白质设计', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/fig_rfdiffusion.png', bbox_inches='tight')
plt.show()
print("RFdiffusion: 从噪声去噪生成蛋白质骨架结构。")""")

md("""## 3. ProteinMPNN

### 3.1 序列设计

```
ProteinMPNN:
  给定骨架结构 → 设计氨基酸序列

  输入: 3D 骨架坐标
  输出: 氨基酸序列

  目标: 序列折叠成给定结构的概率高

架构:
  - 基于图神经网络
  - 残基为节点
  - 距离为边
  - 自回归生成序列
```

### 3.2 为什么需要 MPNN？

```
设计流程:
  RFdiffusion → 骨架结构
  ProteinMPNN → 氨基酸序列
  AlphaFold2 → 验证

为什么不直接用 AF2 逆向?
  - AF2 是序列→结构，不易逆向
  - MPNN 专门训练结构→序列
  - MPNN 成功率更高
```""")

code("""# ProteinMPNN 简化
class ProteinMPNNSimplified(nn.Module):
    def __init__(self, n_amino=20, d_model=64):
        super().__init__()
        self.embed = nn.Linear(3, d_model)
        self.gnn = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.head = nn.Linear(128, n_amino)

    def forward(self, coords):
        # coords: (L, 3) 骨架坐标
        embedded = self.embed(coords)  # (L, d_model)
        features = self.gnn(embedded)  # (L, 128)
        logits = self.head(features)  # (L, 20)
        return logits

    def design(self, coords, temperature=1.0):
        logits = self.forward(coords) / temperature
        probs = F.softmax(logits, dim=-1)
        sequence = torch.argmax(probs, dim=-1)
        return sequence, probs

# 演示
mpnn = ProteinMPNNSimplified()

# 用 RFdiffusion 生成的骨架
sequence, probs = mpnn.design(coords1)

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
print("ProteinMPNN 序列设计:")
print(f"  骨架: {len(coords1)} 个残基")
print(f"  序列: {''.join([amino_acids[s] for s in sequence.numpy()])}")
print(f"  平均置信度: {probs.max(dim=-1)[0].mean():.4f}")""")

md("""## 4. 设计验证

### 4.1 用 AlphaFold2 验证

```
设计验证流程:
  1. RFdiffusion → 骨架
  2. ProteinMPNN → 序列
  3. AlphaFold2(序列) → 预测结构
  4. 比较: 预测结构 vs 设计骨架
  5. 如果一致 → 设计成功!

指标:
  - RMSD: 均方根偏差 (越低越好)
  - pLDDT: 置信度 (越高越好)
  - TM-score: 拓扑匹配 (越高越好)
```""")

code("""# 设计验证
def calculate_rmsd(coords1, coords2):
    # 均方根偏差
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

# 模拟 AF2 预测 (加一些噪声)
np.random.seed(42)
predicted = coords1.numpy() + np.random.randn(*coords1.numpy().shape) * 0.5

rmsd = calculate_rmsd(coords1.numpy(), predicted)
print(f"设计验证:")
print(f"  设计骨架 vs AF2 预测:")
print(f"  RMSD: {rmsd:.3f} Å")
print(f"  {'✅ 设计成功 (RMSD < 2.0)' if rmsd < 2.0 else '❌ 需要重新设计'}")

# 可视化
fig, ax = plt.subplots(figsize=(8, 6))
design = coords1.numpy()
ax.plot(design[:, 0], design[:, 1], 'b-o', markersize=5, linewidth=2, label='设计骨架')
ax.plot(predicted[:, 0], predicted[:, 1], 'r--s', markersize=5, linewidth=2, label='AF2 预测')
ax.set_title(f'设计验证 (RMSD={rmsd:.2f}Å)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
ax.set_xlabel('X'); ax.set_ylabel('Y')

plt.tight_layout()
plt.savefig('notebooks/fig_protein_design_verify.png', bbox_inches='tight')
plt.show()""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 蛋白质设计 (逆向问题) | ✅ |
| RFdiffusion (扩散生成骨架) | ✅ |
| ProteinMPNN (序列设计) | ✅ |
| 设计验证 (AF2 + RMSD) | ✅ |

### 核心 takeaway
> **蛋白质设计 = RFdiffusion + ProteinMPNN + AlphaFold2**——扩散模型生成骨架，MPNN 设计序列，AF2 验证。从预测到设计的飞跃。

### 🔗 下一章
**`65_molecule_pinn.ipynb`** — 分子生成、PINN/神经算子

---

> 💬 **板块十一(AI for Science)进行中 (2/6)。**""")

output_path = "notebooks/64_protein_design.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")