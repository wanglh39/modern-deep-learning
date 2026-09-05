# 生成 67_weather_climate.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 67 — AI 天气预测：GraphCast 与 Pangu

> 🔥 AI 天气预测超越传统数值天气预报——10秒 vs 1小时，且更准。

## 本章你将掌握

1. **数值天气预报**：传统方法
2. **GraphCast**：DeepMind 的图神经网络
3. **Pangu-Weather**：华为的 3D Transformer
4. **AI vs 传统**：速度与精度""")

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

md("""## 1. 数值天气预报 (NWP)

### 1.1 传统方法

```
数值天气预报 (NWP):
  把大气方程离散化 → 在网格上求解

流程:
  1. 收集观测数据
  2. 数据同化 → 初始场
  3. 求解流体力学方程
  4. 输出预报

问题:
  - 计算量大 (超级计算机)
  - 一次预报 ~1-2 小时
  - 分辨率受限
  - 误差累积
```

### 1.2 AI 的优势

```
AI 天气预报:
  从历史数据学习: 当前状态 → 未来状态

优势:
  - 速度快 (10秒 vs 1小时)
  - 不需要解方程
  - 可以学到传统方法没有的模式
  - 普通 GPU 就能跑
```

> 💡 AI 天气预报的核心：从数据学习大气动力学，而非解方程。""")

md("""## 2. GraphCast

### 2.1 DeepMind 的方法

```
GraphCast:
  - 把地球表面建模为图
  - 节点: 网格点
  - 边: 邻接关系
  - 用 GNN 预测

架构:
  1. 输入: 当前大气状态 (多变量)
  2. 编码到图
  3. GNN 消息传递 (多步)
  4. 解码到网格
  5. 输出: 未来状态

特点:
  - 0.25° 分辨率 (~25km)
  - 10 天预报
  - 一次预报 ~10 秒
```""")

code("""# GraphCast 简化
class GraphCastSimplified(nn.Module):
    def __init__(self, n_vars=5, d_model=64, n_layers=4):
        super().__init__()
        self.encoder = nn.Linear(n_vars, d_model)
        self.gnn_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        self.decoder = nn.Linear(d_model, n_vars)

    def message_passing(self, x, adj):
        # 简化的消息传递
        messages = torch.matmul(adj, x)
        return x + messages

    def forward(self, state, adj):
        # state: (B, N, n_vars) N=网格点数
        x = self.encoder(state)  # (B, N, d_model)

        for gnn in self.gnn_layers:
            x = gnn(x)
            x = self.message_passing(x, adj)
            x = F.relu(x)

        output = self.decoder(x)  # (B, N, n_vars)
        return state + output  # 残差预测

# 模拟地球网格
n_grid = 100  # 简化: 100 个网格点
n_vars = 5    # 温度、湿度、风u、风v、气压

# 随机邻接矩阵 (简化)
adj = torch.randn(1, n_grid, n_grid)
adj = (adj + adj.transpose(1, 2)) / 2  # 对称化

model = GraphCastSimplified(n_vars=n_vars)

# 模拟初始状态
state = torch.randn(1, n_grid, n_vars) * 10

# 预报 10 步
predictions = [state.clone()]
with torch.no_grad():
    for step in range(10):
        state = model(state, adj)
        predictions.append(state.clone())

print(f"GraphCast 预报:")
print(f"  网格点: {n_grid}")
print(f"  变量数: {n_vars}")
print(f"  预报步数: {len(predictions)-1}")

# 可视化
fig, axes = plt.subplots(2, 5, figsize=(18, 6))
for i, ax in enumerate(axes.flat):
    if i < len(predictions):
        data = predictions[i][0, :, 0].numpy()  # 温度
        ax.imshow(data.reshape(10, 10), cmap='RdBu_r', vmin=-15, vmax=15)
        ax.set_title(f't={i}', fontsize=11, fontweight='bold')
        ax.axis('off')

plt.suptitle('GraphCast 天气预报 (温度场)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/fig_graphcast.png', bbox_inches='tight')
plt.show()""")

md("""## 3. Pangu-Weather

### 3.1 华为的方法

```
Pangu-Weather:
  - 基于 3D Transformer
  - 把大气看作 3D 体 (经度×纬度×气压层)
  - 3D 自注意力

架构:
  1. 输入: 3D 大气场
  2. 3D Swin Transformer
  3. 输出: 未来场

特点:
  - 3D 空间注意力
  - 分层 (不同气压层)
  - 和 GraphCast 精度相当
```""")

code("""# Pangu-Weather 简化
class PanguWeatherSimplified(nn.Module):
    def __init__(self, n_vars=5, d_model=64, n_levels=5):
        super().__init__()
        self.n_levels = n_levels
        self.embed = nn.Linear(n_vars, d_model)

        # 3D Transformer (简化为 2D + 层间注意力)
        self.spatial_att = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.level_att = nn.MultiheadAttention(d_model, 4, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_vars)

    def forward(self, x):
        # x: (B, n_levels, N, n_vars)
        B, L, N, V = x.shape

        # 嵌入
        x = self.embed(x)  # (B, L, N, d)

        # 空间注意力 (在每个气压层内)
        x_flat = x.reshape(B*L, N, -1)
        att_out, _ = self.spatial_att(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + att_out)
        x = x_flat.reshape(B, L, N, -1)

        # 层间注意力
        x_flat = x.permute(0, 2, 1, 3).reshape(B*N, L, -1)
        att_out, _ = self.level_att(x_flat, x_flat, x_flat)
        x_flat = self.norm2(x_flat + att_out)
        x = x_flat.reshape(B, N, L, -1).permute(0, 2, 1, 3)

        return self.head(x)

# 演示
pangu = PanguWeatherSimplified(n_vars=5, d_model=32, n_levels=5)
state_3d = torch.randn(1, 5, 100, 5)  # (B, levels, grid, vars)

with torch.no_grad():
    pred = pangu(state_3d)

print(f"Pangu-Weather:")
print(f"  输入: {state_3d.shape} (B, 气压层, 网格点, 变量)")
print(f"  输出: {pred.shape}")
print("3D Transformer: 空间注意力 + 层间注意力。")""")

md("""## 4. AI vs 传统方法

### 4.1 对比

```
              传统 NWP      GraphCast    Pangu
速度          ~1 小时       ~10 秒       ~10 秒
精度          基准          更好         相当
分辨率        9-25km        25km         25km
计算资源      超算          GPU          GPU
可解释性      高 (解方程)   低 (黑盒)    低 (黑盒)
```

### 4.2 局限

```
AI 天气预报的局限:
  1. 极端天气: 训练数据少 → 可能不准
  2. 长期预报: 超过 10 天 → 误差大
  3. 可解释性: 黑盒 → 难理解为什么
  4. 数据依赖: 需要大量历史数据
```""")

code("""# AI vs 传统对比
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['传统 NWP', 'GraphCast', 'Pangu-Weather']
speed = [3600, 10, 12]  # 秒
accuracy = [85, 90, 89]  # %
cost = [10, 1, 1]  # 相对成本

x = np.arange(len(methods))
width = 0.25

ax.bar(x - width, [s/60 for s in speed], width, label='时间 (分钟)', color='steelblue', alpha=0.8)
ax.bar(x, [a/100 for a in accuracy], width, label='归一化精度', color='forestgreen', alpha=0.8)
ax.bar(x + width, [c/10 for c in cost], width, label='归一化成本', color='coral', alpha=0.8)

ax.set_xlabel('方法', fontsize=12)
ax.set_ylabel('归一化值', fontsize=12)
ax.set_title('AI vs 传统天气预报', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_weather_comparison.png', bbox_inches='tight')
plt.show()
print("AI 方法: 速度快 360 倍，精度更高，成本更低。")""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 数值天气预报 (NWP) | ✅ |
| GraphCast (GNN) | ✅ |
| Pangu-Weather (3D Transformer) | ✅ |
| AI vs 传统对比 | ✅ |

### 核心 takeaway
> **AI 天气预报超越传统 NWP**——GraphCast 用 GNN，Pangu 用 3D Transformer，速度快 360 倍且更准。但极端天气和可解释性仍是挑战。

### 🔗 下一章
**`68_automated_discovery.ipynb`** — AlphaProof、AlphaEvolve

---

> 💬 **板块十一(AI for Science)进行中 (5/6)。**""")

output_path = "notebooks/67_weather_climate.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")