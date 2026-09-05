# 生成 66_materials_discovery.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 66 — GNoME：AI 材料发现

> 🔥 DeepMind 用 AI 发现了 220 万种新晶体——相当于人类 800 年的发现。

## 本章你将掌握

1. **材料发现问题**：为什么难
2. **GNoME**：图网络材料探索
3. **晶体结构预测**：从元素到结构
4. **应用**：电池、超导、催化剂""")

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

md("""## 1. 材料发现问题

### 1.1 为什么难？

```
材料发现:
  给定元素组合 → 找到稳定的晶体结构

挑战:
  - 搜索空间巨大 (元素组合 × 结构)
  - 稳定性验证需要 DFT 计算 (慢)
  - 大部分组合不稳定

传统方法:
  1. 化学家凭经验猜测
  2. DFT 计算验证
  3. 一次一个 → 慢

AI 方法 (GNoME):
  1. AI 预测可能稳定的结构
  2. 只对高概率的做 DFT
  3. 批量发现 → 快
```

### 1.2 影响力

```
GNoME 成果:
  - 220 万新候选材料
  - 38 万稳定结构
  - 相当于人类 800 年的发现

应用方向:
  - 电池: 更高能量密度
  - 超导: 室温超导
  - 催化剂: 更高效
  - 半导体: 更小更快
```

> 💡 GNoME 是 AI for Science 的里程碑——AI 大幅加速材料发现。""")

md("""## 2. GNoME 方法

### 2.1 核心流程

```
GNoME (Graph Networks for Material Exploration):
  1. 生成候选结构 (替换/变形)
  2. 用 GNN 预测能量
  3. 筛选低能量 (稳定) 候选
  4. DFT 验证
  5. 把验证结果加入训练
  6. 重复 (主动学习)

关键:
  - GNN 预测能量 (快)
  - DFT 验证 (准但慢)
  - 主动学习 (不断改进)
```

### 2.2 稳定性判断

```
凸包 (Convex Hull):
  - 已知稳定材料的能量形成凸包
  - 新材料能量在凸包上方 → 不稳定
  - 新材料能量在凸包上 → 稳定
  - 距凸包的距离 = 稳定性度量

  距离 < 0 (在凸包下) → 稳定
  距离 > 0 (在凸包上) → 不稳定
```""")

code("""# GNoME 简化实现
class MaterialGNN(nn.Module):
    def __init__(self, n_elements=50, d_model=64):
        super().__init__()
        self.element_embed = nn.Embedding(n_elements, d_model)
        self.gnn = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, d_model)
        )
        self.energy_head = nn.Linear(d_model, 1)

    def forward(self, composition):
        # composition: (B, n_atoms) 元素索引
        embeddings = self.element_embed(composition)  # (B, n_atoms, d_model)
        graph_out = self.gnn(embeddings)  # (B, n_atoms, d_model)
        pooled = graph_out.mean(dim=1)  # (B, d_model)
        energy = self.energy_head(pooled)  # (B, 1)
        return energy

    def is_stable(self, composition, hull_energy=0.0):
        energy = self.forward(composition)
        distance = energy.item() - hull_energy
        return distance < 0, distance

# 模拟材料发现
gnn = MaterialGNN(n_elements=50)

# 已知稳定材料的能量 (凸包)
known_stable = {
    'Li2O': -6.0,
    'NaCl': -4.0,
    'SiO2': -8.0,
    'TiO2': -9.0,
}

# 生成候选材料
np.random.seed(42)
candidates = []
for _ in range(100):
    n_atoms = np.random.randint(2, 8)
    comp = torch.randint(0, 20, (1, n_atoms))
    with torch.no_grad():
        energy = gnn(comp).item()
    candidates.append((comp, energy))

# 筛选稳定候选 (能量 < -5.0)
stable = [c for c in candidates if c[1] < -5.0]

print(f"GNoME 材料发现:")
print(f"  候选材料: {len(candidates)}")
print(f"  稳定候选 (能量 < -5.0): {len(stable)}")
print(f"  筛选率: {len(stable)/len(candidates):.1%}")
print(f"\\n  → 只对 {len(stable)} 个做 DFT 验证 (而非 {len(candidates)} 个)")""")

code("""# 可视化材料发现过程
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 能量分布
ax = axes[0]
energies = [e for _, e in candidates]
ax.hist(energies, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(x=-5.0, color='red', linestyle='--', linewidth=2, label='稳定性阈值')
ax.set_xlabel('预测能量 (eV)', fontsize=12)
ax.set_ylabel('候选数', fontsize=12)
ax.set_title('候选材料能量分布', fontsize=13, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)

# 主动学习曲线
ax = axes[1]
iterations = np.arange(1, 11)
discovered = 1000 * iterations**1.5
ax.plot(iterations, discovered, 'b-o', linewidth=2, markersize=8)
ax.set_xlabel('主动学习迭代', fontsize=12)
ax.set_ylabel('累计发现材料数', fontsize=12)
ax.set_title('GNoME 主动学习', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_gnome.png', bbox_inches='tight')
plt.show()
print("左: 大部分候选不稳定，只筛选少数做 DFT; 右: 主动学习不断发现新材料。")""")

md("""## 3. 应用场景

### 3.1 电池材料

```
目标: 更高能量密度的电池材料
  - 正极材料: Li-ion, Na-ion
  - 固态电解质
  - 负极材料

GNoME 发现:
  - 新的锂离子导体
  - 新的钠离子电池材料
  → 可能下一代电池
```

### 3.2 超导材料

```
目标: 室温超导
  - 高温超导机理不明
  - AI 搜索新材料空间
  → 候选超导材料
```""")

md("""## 4. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 材料发现挑战 | ✅ |
| GNoME (GNN + 主动学习) | ✅ |
| 凸包稳定性判断 | ✅ |
| 应用 (电池/超导) | ✅ |

### 核心 takeaway
> **GNoME 用 AI 加速材料发现**——GNN 预测能量，主动学习迭代改进，DFT 验证。220 万新候选材料是 AI for Science 的里程碑。

### 🔗 下一章
**`67_weather_climate.ipynb`** — GraphCast/Pangu 天气预测

---

> 💬 **板块十一(AI for Science)进行中 (4/6)。**""")

output_path = "notebooks/66_materials_discovery.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")