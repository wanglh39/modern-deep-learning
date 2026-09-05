# 生成 63_alphafold.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 63 — AlphaFold：蛋白质结构预测

> AI 解决了生物学50年的难题——从氨基酸序列预测蛋白质三维结构。

## 本章你将掌握

1. **蛋白质结构问题**：为什么重要且难
2. **AlphaFold 演进**：AF1 → AF2 → AF3
3. **EvoFormer**：核心架构
4. **结构模块**：从距离到坐标""")

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

md("""## 1. 蛋白质结构预测问题

### 1.1 为什么重要？

```
蛋白质:
  氨基酸序列 (1D) → 三维结构 (3D) → 功能

  序列: MKTAYIAKQRQISFVKSHFSRLE...
  结构: 折叠成复杂3D形状
  功能: 由3D结构决定

中心法则:
  DNA → RNA → 蛋白质序列 → 蛋白质结构 → 功能

问题: 知道序列，如何预测结构？
  - 实验方法 (X射线晶体学) 慢且贵
  - 计算预测 难 (50年难题)
  → AlphaFold 解决了它!
```

### 1.2 四级结构

```
一级: 氨基酸序列 (线性)
二级: α螺旋、β折叠 (局部)
三级: 整个链的3D折叠
四级: 多链组装

AlphaFold 预测: 三级结构
```

> 💡 蛋白质结构决定功能——知道结构就能理解疾病、设计药物。""")

md("""## 2. AlphaFold 演进

### 2.1 从 AF1 到 AF3

```
AlphaFold 1 (2018):
  - 用深度学习预测距离分布
  - 再用梯度优化结构
  - CASP13: 首次突破

AlphaFold 2 (2020):
  - 端到端: 序列 → 结构
  - EvoFormer + 结构模块
  - CASP14: 接近实验精度
  - 革命性突破!

AlphaFold 3 (2024):
  - 不只蛋白质: DNA/RNA/配体
  - 更精确的相互作用
  - 扩散模块替代结构模块
```

### 2.2 AlphaFold 2 的核心

```
AlphaFold 2 流程:
  1. 输入: 氨基酸序列
  2. MSA: 多序列比对 (找同源序列)
  3. EvoFormer: 进化信息 + 结构信息
  4. 结构模块: → 3D 坐标
  5. 输出: 原子坐标 + 置信度

关键创新:
  - EvoFormer: 联合处理 MSA 和 pair 表示
  - 端到端: 直接输出坐标
  - 不需要后处理优化
```""")

code("""# AlphaFold 2 简化架构
class AlphaFold2Simplified(nn.Module):
    def __init__(self, n_residues=50, d_msa=64, d_pair=64, n_heads=4):
        super().__init__()
        self.n_residues = n_residues

        # MSA 表示 (序列对齐信息)
        self.msa_embed = nn.Linear(21, d_msa)  # 20 氨基酸 + gap

        # Pair 表示 (残基对信息)
        self.pair_embed = nn.Linear(1, d_pair)

        # EvoFormer (简化)
        self.msa_att = nn.MultiheadAttention(d_msa, n_heads, batch_first=True)
        self.pair_att = nn.MultiheadAttention(d_pair, n_heads, batch_first=True)
        self.msa_norm = nn.LayerNorm(d_msa)
        self.pair_norm = nn.LayerNorm(d_pair)

        # 结构模块
        self.structure_module = nn.Linear(d_pair, 3)  # → 3D 坐标

    def forward(self, sequence):
        # sequence: (B, L) 氨基酸索引
        B, L = sequence.shape

        # One-hot 编码
        msa = F.one_hot(sequence, 21).float()  # (B, L, 21)
        msa = self.msa_embed(msa)  # (B, L, d_msa)

        # Pair 表示 (初始化为距离)
        pair = torch.zeros(B, L, L, 1)
        pair = self.pair_embed(pair)  # (B, L, L, d_pair)

        # EvoFormer (简化: 1 层)
        msa_out, _ = self.msa_att(msa, msa, msa)
        msa = self.msa_norm(msa + msa_out)

        # 结构模块: 从 pair 表示预测坐标
        pair_flat = pair.mean(dim=2)  # (B, L, d_pair)
        coords = self.structure_module(pair_flat)  # (B, L, 3)

        return coords

# 模拟蛋白质
n_res = 30
sequence = torch.randint(0, 20, (1, n_res))  # 随机氨基酸序列

model = AlphaFold2Simplified(n_residues=n_res)
with torch.no_grad():
    coords = model(sequence)[0].numpy()

print(f"蛋白质: {n_res} 个残基")
print(f"序列: {sequence[0].numpy()[:10]}... (前10个)")
print(f"坐标形状: {coords.shape}")
print(f"坐标范围: x=[{coords[:,0].min():.2f}, {coords[:,0].max():.2f}], "
      f"y=[{coords[:,1].min():.2f}, {coords[:,1].max():.2f}], "
      f"z=[{coords[:,2].min():.2f}, {coords[:,2].max():.2f}]")""")

code("""# 可视化蛋白质结构
fig = plt.figure(figsize=(12, 5))

# 3D 结构
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'b-o', markersize=4, linewidth=2)
ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=range(n_res),
           cmap='viridis', s=50)
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_title('蛋白质 3D 结构', fontsize=13, fontweight='bold')

# 距离矩阵
ax2 = fig.add_subplot(122)
dist_matrix = np.zeros((n_res, n_res))
for i in range(n_res):
    for j in range(n_res):
        dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

im = ax2.imshow(dist_matrix, cmap='viridis_r')
ax2.set_xlabel('残基 i'); ax2.set_ylabel('残基 j')
ax2.set_title('距离矩阵', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax2, label='距离 (Å)')

plt.tight_layout()
plt.savefig('notebooks/fig_alphafold.png', bbox_inches='tight')
plt.show()
print("左: 3D 结构 (每个点是残基); 右: 残基间距离矩阵。")""")

md("""## 3. EvoFormer

### 3.1 核心架构

```
EvoFormer: 联合处理 MSA 和 Pair 表示

MSA 表示:
  - 多序列比对结果
  - 包含进化信息
  - 形状: (s, r, d) s=序列数, r=残基数

Pair 表示:
  - 残基对的信息
  - 包含距离/角度
  - 形状: (r, r, d)

信息流:
  MSA → Pair (从对齐推断距离)
  Pair → MSA (从结构约束对齐)
  → 交替更新
```

### 3.2 EvoFormer 模块

```
EvoFormer 包含:
  1. MSA 行注意力 (在序列维度)
  2. MSA 列注意力 (在残基维度)
  3. Pair 三角注意力
  4. Pair 三角乘法
  5. Pair 三角不等式

三角更新:
  - 保证距离满足三角不等式
  - i-j 距离 ≤ i-k 距离 + k-j 距离
  → 物理约束嵌入架构
```

> 💡 EvoFormer 的核心：MSA 和 Pair 表示交替更新，三角约束保证物理合理性。""")

code("""# EvoFormer 简化
class EvoFormerSimplified(nn.Module):
    def __init__(self, d_msa=64, d_pair=64, n_heads=4):
        super().__init__()
        # MSA 模块
        self.msa_row_att = nn.MultiheadAttention(d_msa, n_heads, batch_first=True)
        self.msa_col_att = nn.MultiheadAttention(d_msa, n_heads, batch_first=True)
        self.msa_norm1 = nn.LayerNorm(d_msa)
        self.msa_norm2 = nn.LayerNorm(d_msa)

        # Pair 模块
        self.pair_att = nn.MultiheadAttention(d_pair, n_heads, batch_first=True)
        self.pair_norm = nn.LayerNorm(d_pair)

        # MSA → Pair 通信
        self.msa_to_pair = nn.Linear(d_msa, d_pair)
        # Pair → MSA 通信
        self.pair_to_msa = nn.Linear(d_pair, d_msa)

    def forward(self, msa, pair):
        # msa: (B, s, r, d_msa) 简化为 (B, r, d_msa)
        # pair: (B, r, r, d_pair) 简化为 (B, r, d_pair)

        # MSA 更新
        msa_att_out, _ = self.msa_row_att(msa, msa, msa)
        msa = self.msa_norm1(msa + msa_att_out)

        # Pair → MSA
        msa = msa + self.pair_to_msa(pair.mean(dim=1))

        # Pair 更新
        pair_att_out, _ = self.pair_att(pair, pair, pair)
        pair = self.pair_norm(pair + pair_att_out)

        # MSA → Pair
        pair = pair + self.msa_to_pair(msa.mean(dim=1, keepdim=True))

        return msa, pair

# 演示
evo = EvoFormerSimplified()
msa = torch.randn(1, 30, 64)
pair = torch.randn(1, 30, 64)

for i in range(3):
    msa, pair = evo(msa, pair)
    print(f"EvoFormer 层 {i+1}: MSA 范数={msa.norm():.2f}, Pair 范数={pair.norm():.2f}")

print("\\nEvoFormer: MSA 和 Pair 交替更新，信息双向流动。")""")

md("""## 4. AlphaFold 3

### 4.1 新能力

```
AlphaFold 3 (2024):
  不只蛋白质，还能预测:
  - 蛋白质-DNA 相互作用
  - 蛋白质-RNA 相互作用
  - 蛋白质-配体 结合
  - DNA-RNA 杂交

架构变化:
  - 用扩散模型替代结构模块
  - 更统一的输入处理
  - 更精确的相互作用

影响:
  - 药物设计: 预测药物-靶点结合
  - 疾病理解: 理解突变如何影响结构
  - 合成生物学: 设计新蛋白
```""")

code("""# AlphaFold 3 的扩散结构模块
class DiffusionStructureModule(nn.Module):
    def __init__(self, d_model=64, n_steps=50):
        super().__init__()
        self.n_steps = n_steps
        self.net = nn.Sequential(
            nn.Linear(d_model + 3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, features, coords_init=None):
        # features: (B, L, d_model)
        B, L, D = features.shape

        if coords_init is None:
            coords = torch.randn(B, L, 3) * 5  # 初始噪声
        else:
            coords = coords_init

        trajectory = [coords.clone()]

        # 反向扩散
        for t in range(self.n_steps):
            # 预测噪声
            inp = torch.cat([features, coords], dim=-1)
            noise_pred = self.net(inp)

            # 去噪步
            dt = 1.0 / self.n_steps
            coords = coords - noise_pred * dt
            trajectory.append(coords.clone())

        return coords, trajectory

# 演示
diffusion = DiffusionStructureModule(d_model=64, n_steps=30)
features = torch.randn(1, 20, 64)

with torch.no_grad():
    final_coords, trajectory = diffusion(features)

print(f"AlphaFold 3 扩散结构模块:")
print(f"  初始: 随机噪声 (坐标范围: {trajectory[0].std():.2f})")
print(f"  最终: 结构 (坐标范围: {final_coords.std():.2f})")
print(f"  步数: {len(trajectory)}")

# 可视化扩散过程
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
steps_to_show = [0, 10, 20, 29]
for ax, step in zip(axes, steps_to_show):
    coords = trajectory[step][0].numpy()
    ax.scatter(coords[:, 0], coords[:, 1], c=range(len(coords)),
              cmap='viridis', s=30)
    ax.set_title(f'步 {step}', fontsize=12, fontweight='bold')
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.grid(alpha=0.3)

plt.suptitle('AlphaFold 3 扩散: 从噪声到结构', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/fig_alphafold3.png', bbox_inches='tight')
plt.show()""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 蛋白质结构预测问题 | ✅ |
| AlphaFold 2 (EvoFormer + 结构模块) | ✅ |
| EvoFormer (MSA + Pair 交替) | ✅ |
| AlphaFold 3 (扩散 + 多分子) | ✅ |

### 核心 takeaway
> **AlphaFold 解决了50年难题**——EvoFormer 联合进化信息和结构约束，端到端预测3D坐标。AF3 用扩散模型扩展到多分子相互作用。

### 🔗 下一章
**`64_protein_design.ipynb`** — RFdiffusion、蛋白质设计

---

> 💬 **板块十一(AI for Science)进行中 (1/6)。**""")

output_path = "notebooks/63_alphafold.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")