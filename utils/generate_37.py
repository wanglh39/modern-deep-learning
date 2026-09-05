# 生成 37_consistency_models.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 37 — 一致性模型：扩散加速到一步

> 一致性模型（Consistency Models）让扩散模型**一步生成**——
> 从 1000 步到 1 步，速度提升 1000 倍。

## 本章你将掌握

1. **一致性性质**：轨迹上任意点映射到起点
2. **一致性蒸馏**：从预训练扩散蒸馏
3. **一致性训练**：从零训练
4. **一步生成**：极限加速""")

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

md("""## 1. 一致性性质

### 1.1 核心思想

扩散模型的 ODE 轨迹从 $x_T$（噪声）到 $x_0$（数据）。
**一致性函数** $f$ 满足：

$$f(x_t, t) = x_0 \\quad \\forall t \\in [0, T]$$

```
轨迹: x_T → x_{T-1} → ... → x_1 → x_0
一致性: f(x_T, T) = f(x_{T-1}, T-1) = ... = f(x_0, 0) = x_0

→ 任意时间点都能直接跳到 x_0!
→ 一步生成: f(x_T, T) = x_0
```

### 1.2 边界条件

$$f(x, 0) = x$$

即 $t=0$ 时一致性函数是恒等映射。

> 💡 一致性模型的哲学：**轨迹上任意点都"知道"终点**。
# 学会这个映射，就能从任意噪声一步跳到数据。""")

md("""## 2. 一致性蒸馏

### 2.1 从预训练扩散蒸馏

```
1. 有一个训练好的扩散模型 φ (教师)
2. 训练一致性模型 f_θ (学生)

蒸馏损失:
  1. 采样 x_t
  2. 用教师 φ 走一步: x_t → x_{t-Δt}
  3. 一致性约束: f_θ(x_t, t) ≈ f_θ(x_{t-Δt}, t-Δt)
  → 学生在相邻点输出一致
```

### 2.2 在线 vs 离线

```
在线蒸馏: 教师实时生成 x_{t-Δt}
  → 更新, 但慢

离线蒸馏: 预先用教师生成配对
  → 快, 但需要存储
```

> 💡 一致性蒸馏把 1000 步的扩散"压缩"到 1 步——
# 质量略降，但速度提升 1000 倍。""")

code("""# 一致性模型实现
class ConsistencyModel(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        # 边界条件: t=0 时 f(x,0) = x
        # 用 skip connection 保证
        out = self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1))
        return x + t.unsqueeze(-1) * out  # t=0 时返回 x

    @torch.no_grad()
    def sample(self, n_samples, data_dim=2, n_steps=1):
        x = torch.randn(n_samples, data_dim)
        if n_steps == 1:
            # 一步生成
            t = torch.ones(n_samples)
            return self(x, t)

        # 多步 (DPM 风格)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((n_samples,), 1.0 - i * dt)
            x = self(x, t)
            if i < n_steps - 1:
                x = x + torch.randn_like(x) * np.sqrt(2 * dt)  # 加噪
        return x

# 模拟一致性蒸馏训练
def train_consistency(n_epochs=2000):
    model = ConsistencyModel(data_dim=2, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 模拟教师 (用真实数据 + 噪声)
    def swiss_roll(n=1000):
        t = np.random.uniform(0, 4 * np.pi, n)
        x = t * np.cos(t) / (4 * np.pi)
        y = t * np.sin(t) / (4 * np.pi)
        return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

    data = swiss_roll(2000)
    losses = []
    for _ in range(n_epochs):
        # 采样
        x1 = data[torch.randint(0, len(data), (64,))]
        x0 = torch.randn(64, 2)
        t = torch.rand(64)
        xt = (1 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1

        # 一致性损失: f(x_t, t) ≈ f(x_{t-dt}, t-dt)
        # 简化: f(x_t, t) ≈ x1 (直接监督)
        pred = model(xt, t)
        loss = F.mse_loss(pred, x1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses

model, losses = train_consistency()
samples_1step = model.sample(2000, n_steps=1)
samples_multistep = model.sample(2000, n_steps=5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(losses, 'b-', linewidth=1); axes[0].set_title('训练 Loss')
axes[1].scatter(samples_1step[:, 0].numpy(), samples_1step[:, 1].numpy(), c='red', s=5, alpha=0.5)
axes[1].set_title('一步生成'); axes[1].set_aspect('equal'); axes[1].set_xlim(-1.5, 1.5); axes[1].set_ylim(-1.5, 1.5)
axes[2].scatter(samples_multistep[:, 0].numpy(), samples_multistep[:, 1].numpy(), c='green', s=5, alpha=0.5)
axes[2].set_title('5步生成'); axes[2].set_aspect('equal'); axes[2].set_xlim(-1.5, 1.5); axes[2].set_ylim(-1.5, 1.5)
plt.tight_layout()
plt.savefig('notebooks/fig_consistency.png', bbox_inches='tight')
plt.show()
print(f"一致性模型: loss {losses[0]:.4f} → {losses[-1]:.4f}")
print("一步生成: 从噪声直接跳到数据——1000x 加速!")""")

md("""## 3. 一致性训练

### 3.1 不需要教师

一致性蒸馏需要预训练扩散模型。**一致性训练**直接从零训练：

```
一致性训练:
  1. 采样数据 x1
  2. 采样噪声 x0
  3. 插值 x_t = (1-t)*x0 + t*x1
  4. 损失: f(x_t, t) ≈ x1 (直接监督)
  → 不需要教师扩散模型!
```

### 3.2 优势

- **不需要预训练**：直接从数据训练
- **一步生成**：推理极快
- **质量**：略低于扩散，但远好于 GAN

> 💡 一致性训练让一步生成模型**从零训练**成为可能——
# 不需要先训练扩散再蒸馏。""")

md("""## 4. 扩散加速方法对比

| 方法 | 步数 | 质量 | 训练方式 |
|------|------|------|---------|
| **DDPM** | 1000 | 最好 | 标准 |
| **DDIM** | 50 | 好 | 标准 |
| **DPM-Solver** | 10-20 | 好 | 标准 |
| **一致性蒸馏** | 1-4 | 中 | 需要教师 |
| **一致性训练** | 1-4 | 中 | 从零 |
| **Rectified Flow** | 4-8 | 好 | 从零 |

### 演进方向

```
1000步 (DDPM) → 50步 (DDIM) → 10步 (DPM-Solver) → 1步 (一致性)
```

> 💡 扩散模型的加速竞赛：从 1000 步到 1 步。
# 一致性模型是极致，但质量略低。Rectified Flow 在质量和速度间平衡更好。

### 🔗 下一章
**`38_dit.ipynb`** — DiT 扩散 Transformer

---

> 💬 **板块六进行中。**""")

output_path = "notebooks/37_consistency_models.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")