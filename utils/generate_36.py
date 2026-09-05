# 生成 36_flow_matching.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 36 — Flow Matching：连续归一化流

> Flow Matching 是扩散模型的推广——学习从噪声到数据的**连续流**。
> Rectified Flow 让路径更直，生成更快。Stable Diffusion 3 用了它。

## 本章你将掌握

1. **Flow Matching 原理**：ODE 和向量场
2. **Rectified Flow**：拉直路径
3. **与扩散的关系**：扩散是特例
4. **优势**：更直路径 → 更快生成""")

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
plt.rcParams['figure.dpi'] = 120; plt.rcParams['font.size'] = 11
np.random.seed(42); torch.manual_seed(42)
print("环境准备完成 ✅")""")

md("""## 1. Flow Matching 原理

### 1.1 核心思想

不学离散的去噪步骤，而是学一个**连续的流**：

```
x(0) = 噪声 (先验分布)
x(1) = 数据 (目标分布)

学一个向量场 v(x, t):
  dx/dt = v(x, t)
  → 从 t=0 积分到 t=1, 把噪声变成数据
```

### 1.2 训练目标

Flow Matching 的损失：

$$\\mathcal{L} = \\mathbb{E}_{t, x_0, x_1} \\| v_\\theta(x_t, t) - (x_1 - x_0) \\|^2$$

其中 $x_t = (1-t) x_0 + t x_1$（线性插值）。

```
训练:
1. 采样数据 x1
2. 采样噪声 x0
3. 随机时间 t ∈ [0, 1]
4. 插值 x_t = (1-t)*x0 + t*x1
5. 目标: v = x1 - x0 (从噪声指向数据)
6. 损失: MSE(v_θ(x_t, t), x1 - x0)
```

> 💡 Flow Matching 比 DDPM 更直观——直接学从噪声到数据的"速度场"。
# 沿着这个场积分，就把噪声变成数据。""")

code("""# Flow Matching 实现
class FlowMatching(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))

    def loss(self, x1):
        batch_size = x1.shape[0]
        t = torch.rand(batch_size)
        x0 = torch.randn_like(x1)
        xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        target = x1 - x0  # 目标速度
        pred = self(xt, t)
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample(self, n_samples, data_dim=2, n_steps=50):
        x = torch.randn(n_samples, data_dim)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((n_samples,), i * dt)
            v = self(x, t)
            x = x + v * dt  # Euler 积分
        return x

# 训练 Flow Matching (瑞士卷)
def swiss_roll(n=1000):
    t = np.random.uniform(0, 4 * np.pi, n)
    x = t * np.cos(t) / (4 * np.pi)
    y = t * np.sin(t) / (4 * np.pi)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

data = swiss_roll(2000)
fm = FlowMatching(data_dim=2, hidden_dim=128)
optimizer = torch.optim.Adam(fm.parameters(), lr=1e-3)

losses = []
for _ in range(3000):
    loss = fm.loss(data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

samples = fm.sample(2000, n_steps=50)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), c='blue', s=5, alpha=0.5)
axes[0].set_title('真实数据'); axes[0].set_aspect('equal'); axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)
axes[1].plot(losses, 'b-', linewidth=1); axes[1].set_title('训练 Loss'); axes[1].set_xlabel('Epoch')
axes[2].scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), c='red', s=5, alpha=0.5)
axes[2].set_title('Flow Matching 生成'); axes[2].set_aspect('equal'); axes[2].set_xlim(-1.5, 1.5); axes[2].set_ylim(-1.5, 1.5)
plt.tight_layout()
plt.savefig('notebooks/fig_flow_matching.png', bbox_inches='tight')
plt.show()
print(f"Flow Matching: loss {losses[0]:.4f} → {losses[-1]:.4f}, 50步生成。")""")

md("""## 2. Rectified Flow：拉直路径

### 2.1 路径弯曲的问题

```
Flow Matching: x_t = (1-t)*x0 + t*x1 (线性插值)
  → 路径是直线, 但向量场可能复杂
  → 积分需要小步长

Rectified Flow: 重新拉直
  1. 训练一个 Flow Matching
  2. 用它生成 (x0, x1) 配对
  3. 在这些配对上重新训练 → 路径更直
  → 可以用更少步数生成
```

### 2.2 1-Rectified Flow

```
步骤:
1. 训练 v1 (初始 Flow Matching)
2. 用 v1 生成大量 (x0, x1) 配对
3. 在这些配对上训练 v2
   → v2 的路径更接近直线
4. 可以重复 (2-Rectified, 3-Rectified...)
```

### 2.3 效果

```
普通 Flow Matching: 50-100 步
1-Rectified Flow: 10-20 步
2-Rectified Flow: 5-10 步
→ 路径越直, 步数越少
```

> 💡 Rectified Flow 的思想：**拉直路径**让积分更容易。
# Stable Diffusion 3 用 Rectified Flow 实现了 4-8 步生成。""")

code("""# Rectified Flow: 可视化路径
def visualize_flow_paths(fm_model, n_samples=5, n_steps=50):
    with torch.no_grad():
        x0 = torch.randn(n_samples, 2)
        paths = [x0.numpy()]
        x = x0.clone()
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((n_samples,), i * dt)
            v = fm_model(x, t)
            x = x + v * dt
            paths.append(x.numpy())
        return np.array(paths)

paths = visualize_flow_paths(fm, n_samples=20, n_steps=50)

fig, ax = plt.subplots(figsize=(8, 8))
for i in range(20):
    ax.plot(paths[:, i, 0], paths[:, i, 1], 'b-', alpha=0.3, linewidth=0.5)
    ax.scatter(paths[0, i, 0], paths[0, i, 1], c='red', s=20, zorder=5)  # 起点
    ax.scatter(paths[-1, i, 0], paths[-1, i, 1], c='blue', s=20, zorder=5)  # 终点
ax.set_title('Flow Matching 路径: 噪声(红) → 数据(蓝)')
ax.set_aspect('equal'); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
plt.tight_layout()
plt.savefig('notebooks/fig_flow_paths.png', bbox_inches='tight')
plt.show()
print("Flow Matching 路径: 从噪声(红)到数据(蓝)的连续流。")
print("Rectified Flow: 拉直这些路径 → 更少步数生成。")""")

md("""## 3. Flow Matching vs DDPM

### 3.1 统一视角

```
DDPM:     离散时间, 学噪声 ε
Flow Matching: 连续时间, 学速度 v

关系: DDPM 是 Flow Matching 的特例
  DDPM 的反向过程 ≈ 特定的 ODE
  Flow Matching 更一般, 可以选不同路径
```

### 3.2 优势对比

| 特性 | DDPM | Flow Matching |
|------|------|---------------|
| **路径** | 固定 (由 β 调度决定) | 可选 (线性/弯曲) |
| **步数** | 50-1000 | 4-50 (Rectified) |
| **理论** | SDE | ODE |
| **采样** | 随机 | 确定性 |
| **扩展性** | 较难 | 容易 |

### 3.3 SD3 的选择

Stable Diffusion 3 用 **Rectified Flow**：
- 从 50 步降到 4-8 步
- 质量不降甚至更好
- 训练更简单

> 💡 Flow Matching 正在取代 DDPM 成为扩散模型的新标准。
# SD3, Flux 都转向了 Flow Matching。""")

code("""# DDPM vs Flow Matching 步数对比
steps = [4, 8, 20, 50, 100, 200, 1000]
ddpm_quality = [20, 35, 60, 80, 88, 92, 95]  # 模拟质量 (%)
fm_quality = [75, 88, 93, 95, 96, 97, 98]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, ddpm_quality, 'r-o', linewidth=2, markersize=6, label='DDPM')
ax.plot(steps, fm_quality, 'b-s', linewidth=2, markersize=6, label='Flow Matching')
ax.set_xlabel('采样步数'); ax.set_ylabel('生成质量 (%)')
ax.set_title('DDPM vs Flow Matching: 步数-质量权衡')
ax.set_xscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_fm_vs_ddpm.png', bbox_inches='tight')
plt.show()
print("Flow Matching: 少步质量更好——路径更直, 积分更准。")""")

md("""## 4. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| Flow Matching 原理 | ✅ |
| 向量场与 ODE | ✅ |
| Rectified Flow | ✅ |
| 与 DDPM 的关系 | ✅ |

### 核心 takeaway
> **Flow Matching 学连续流，Rectified Flow 拉直路径**。
# 从 1000 步到 4 步——生成速度提升 250 倍。
# SD3/Flux 都转向 Flow Matching，这是未来方向。

### 🔗 下一章
**`37_consistency_models.ipynb`** — 一致性模型

---

> 💬 **板块六进行中。**""")

output_path = "notebooks/36_flow_matching.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")