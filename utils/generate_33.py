# 生成 33_ddpm.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 33 — DDPM：去噪扩散概率模型

> 扩散模型是当前最强的生成模型。DDPM 从噪声一步步去噪生成图像，
> DDIM 让这个过程可以加速。

## 本章你将掌握

1. **扩散过程**：前向加噪 + 反向去噪
2. **DDPM 训练**：学习去噪网络
3. **DDIM 采样**：确定性 + 加速
4. **从零实现**：迷你 DDPM""")

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

md("""## 1. 扩散过程：前向与反向

### 1.1 前向过程（加噪）

从数据 $x_0$ 逐步加高斯噪声：

$$q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t} x_{t-1}, \\beta_t I)$$

关键性质——**任意步直接采样**：

$$x_t = \\sqrt{\\bar{\\alpha}_t} x_0 + \\sqrt{1-\\bar{\\alpha}_t} \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I)$$

其中 $\\alpha_t = 1 - \\beta_t$, $\\bar{\\alpha}_t = \\prod_{s=1}^{t} \\alpha_s$。

### 1.2 反向过程（去噪）

从纯噪声 $x_T$ 逐步去噪：

$$p_\\theta(x_{t-1} | x_t) = \\mathcal{N}(x_{t-1}; \\mu_\\theta(x_t, t), \\sigma_t^2 I)$$

**训练目标**：学习 $\\mu_\\theta$（或等价地学噪声 $\\epsilon_\\theta$）。

> 💡 DDPM 的优雅：前向加噪有闭式解，训练只需学一个去噪网络。
# 损失函数出人意料地简单——就是 MSE。""")

code("""# 前向过程实现
class DiffusionScheduler:
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.n_steps = n_steps
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise=None):
        # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.alpha_bars[t].sqrt().view(-1, 1)
        sqrt_1mab = (1 - self.alpha_bars[t]).sqrt().view(-1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

# 可视化前向过程
scheduler = DiffusionScheduler(n_steps=1000)
x0 = torch.randn(5, 2) * 0.3 + torch.tensor([2.0, -1.0])  # 数据分布

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
steps = [0, 100, 300, 600, 999]
for ax, t in zip(axes, steps):
    t_tensor = torch.full((5,), t)
    xt, _ = scheduler.add_noise(x0, t_tensor)
    ax.scatter(xt[:, 0].numpy(), xt[:, 1].numpy(), c='blue', s=30)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_title(f't={t}'); ax.set_aspect('equal')
plt.suptitle('前向扩散: 数据 → 噪声', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/fig_forward_diffusion.png', bbox_inches='tight')
plt.show()
print("前向过程: 数据逐步变成纯噪声。")""")

md("""## 2. DDPM 训练

### 2.1 损失函数

DDPM 的训练损失出人意料地简单：

$$\\mathcal{L} = \\mathbb{E}_{t, x_0, \\epsilon} \\left[ \\| \\epsilon - \\epsilon_\\theta(x_t, t) \\|^2 \\right]$$

```
训练步骤:
1. 采样数据 x0
2. 随机采样时间 t
3. 采样噪声 ε
4. 计算 x_t = √(ᾱ_t) x0 + √(1-ᾱ_t) ε
5. 预测噪声 ε_θ(x_t, t)
6. 损失 = MSE(ε, ε_θ)
```

### 2.2 为什么学噪声

学噪声 $\\epsilon_\\theta$ 等价于学均值 $\\mu_\\theta$，但更稳定：

$$\\mu_\\theta(x_t, t) = \\frac{1}{\\sqrt{\\alpha_t}} \\left( x_t - \\frac{\\beta_t}{\\sqrt{1-\\bar{\\alpha}_t}} \\epsilon_\\theta(x_t, t) \\right)$$

> 💡 DDPM 训练就像训练一个"去噪自编码器"——
# 给加噪的图片，预测噪声。简单、稳定、有效。""")

code("""# DDPM 完整实现
class DDPM(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128, n_steps=1000):
        super().__init__()
        self.scheduler = DiffusionScheduler(n_steps)
        self.n_steps = n_steps

        # 去噪网络: (x_t, t) → ε
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def predict_noise(self, x, t):
        # t 归一化到 [0, 1]
        t_norm = t.float().view(-1, 1) / self.n_steps
        return self.net(torch.cat([x, t_norm], dim=-1))

    def training_loss(self, x0):
        batch_size = x0.shape[0]
        # 随机时间
        t = torch.randint(0, self.n_steps, (batch_size,))
        # 随机噪声
        noise = torch.randn_like(x0)
        # 加噪
        xt, _ = self.scheduler.add_noise(x0, t, noise)
        # 预测噪声
        pred_noise = self.predict_noise(xt, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, n_samples, data_dim=2):
        # 从纯噪声开始
        x = torch.randn(n_samples, data_dim)
        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((n_samples,), t)
            eps = self.predict_noise(x, t_tensor)
            alpha = self.scheduler.alphas[t]
            alpha_bar = self.scheduler.alpha_bars[t]
            beta = self.scheduler.betas[t]

            mean = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_bar).sqrt()) * eps)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + beta.sqrt() * noise
            else:
                x = mean
        return x

# 训练 DDPM (2D 瑞士卷数据)
def swiss_roll(n=1000):
    t = np.random.uniform(0, 4 * np.pi, n)
    x = t * np.cos(t) / (4 * np.pi)
    y = t * np.sin(t) / (4 * np.pi)
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

data = swiss_roll(2000)
ddpm = DDPM(data_dim=2, hidden_dim=128, n_steps=200)
optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-3)

losses = []
for epoch in range(2000):
    loss = ddpm.training_loss(data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses, 'b-', linewidth=1)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('DDPM 训练'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_ddpm_train.png', bbox_inches='tight')
plt.show()
print(f"训练完成: loss {losses[0]:.4f} → {losses[-1]:.4f}")""")

code("""# 生成样本
samples = ddpm.sample(2000, data_dim=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), c='blue', s=5, alpha=0.5)
axes[0].set_title('真实数据 (瑞士卷)'); axes[0].set_aspect('equal')
axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)

axes[1].scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), c='red', s=5, alpha=0.5)
axes[1].set_title('DDPM 生成'); axes[1].set_aspect('equal')
axes[1].set_xlim(-1.5, 1.5); axes[1].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('notebooks/fig_ddpm_samples.png', bbox_inches='tight')
plt.show()
print("DDPM 生成的样本逼近真实数据分布!")""")

md("""## 3. DDIM：确定性采样

### 3.1 DDPM 的问题

DDPM 采样需要 **1000 步**，每步一次网络前向——慢。

### 3.2 DDIM 的解决方案

DDIM（Denoising Diffusion Implicit Models）：
- **确定性**：去掉随机噪声 → 同一噪声总是生成同一样本
- **加速**：可以跳过中间步 → 50 步就能生成

### 3.3 DDIM 采样公式

$$x_{t-1} = \\sqrt{\\bar{\\alpha}_{t-1}} \\hat{x}_0 + \\sqrt{1 - \\bar{\\alpha}_{t-1}} \\epsilon_\\theta(x_t, t)$$

其中 $\\hat{x}_0 = \\frac{x_t - \\sqrt{1-\\bar{\\alpha}_t} \\epsilon_\\theta}{\\sqrt{\\bar{\\alpha}_t}}$ 是预测的 $x_0$。

> 💡 DDIM 的关键：**跳过步骤**时不改变生成分布。
# 50 步 DDIM ≈ 1000 步 DDPM 的质量——20 倍加速。""")

code("""# DDIM 采样
class DDIMSampler:
    def __init__(self, ddpm):
        self.ddpm = ddpm
        self.scheduler = ddpm.scheduler

    @torch.no_grad()
    def sample(self, n_samples, data_dim=2, n_steps=50):
        # 选子序列
        step_ratio = self.scheduler.n_steps // n_steps
        timesteps = list(range(0, self.scheduler.n_steps, step_ratio))

        x = torch.randn(n_samples, data_dim)
        for i in reversed(range(len(timesteps))):
            t = timesteps[i]
            t_tensor = torch.full((n_samples,), t)
            eps = self.ddpm.predict_noise(x, t_tensor)

            alpha_bar_t = self.scheduler.alpha_bars[t]
            if i > 0:
                alpha_bar_prev = self.scheduler.alpha_bars[timesteps[i-1]]
            else:
                alpha_bar_prev = torch.tensor(1.0)

            # 预测 x0
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
            # DDIM 步
            x = alpha_bar_prev.sqrt() * x0_pred + (1 - alpha_bar_prev).sqrt() * eps

        return x

# 对比 DDPM vs DDIM
ddim = DDIMSampler(ddpm)

import time
start = time.time()
ddpm_samples = ddpm.sample(1000, data_dim=2)
ddpm_time = time.time() - start

start = time.time()
ddim_samples = ddim.sample(1000, data_dim=2, n_steps=50)
ddim_time = time.time() - start

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(ddpm_samples[:, 0].numpy(), ddpm_samples[:, 1].numpy(), c='blue', s=5, alpha=0.5)
axes[0].set_title(f'DDPM (200步, {ddpm_time:.1f}s)')
axes[0].set_aspect('equal'); axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)

axes[1].scatter(ddim_samples[:, 0].numpy(), ddim_samples[:, 1].numpy(), c='red', s=5, alpha=0.5)
axes[1].set_title(f'DDIM (50步, {ddim_time:.1f}s)')
axes[1].set_aspect('equal'); axes[1].set_xlim(-1.5, 1.5); axes[1].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('notebooks/fig_ddim.png', bbox_inches='tight')
plt.show()
print(f"DDPM: {ddpm_time:.2f}s (200步)")
print(f"DDIM: {ddim_time:.2f}s (50步)")
print(f"加速: {ddpm_time/ddim_time:.1f}x, 质量相当!")""")

md("""## 4. 扩散模型家族

### 4.1 演进路线

```
DDPM (2020)     → 基础扩散模型
DDIM (2020)     → 加速采样
LDM (2022)      → 潜空间扩散 (Stable Diffusion)
DPM-Solver (2022) → 高阶 ODE 求解
Consistency (2023) → 一步生成
Flow Matching (2023) → 连续归一化流
```

### 4.2 关键改进

| 改进 | 效果 |
|------|------|
| **潜空间扩散** | 降低维度, 加快训练 |
| **DPM-Solver** | 10-20 步高质量采样 |
| **一致性模型** | 1-4 步生成 |
| **Flow Matching** | 更直的路径, 更快 |

> 💡 扩散模型的进化方向：**更快、更可控、更高质量**。
# 从 1000 步到 1 步，速度提升 1000 倍。

### 🔗 下一章
**`34_stable_diffusion.ipynb`** — Latent Diffusion、条件生成

---

> 💬 **板块六进行中。**""")

output_path = "notebooks/33_ddpm.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")