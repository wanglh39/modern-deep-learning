# 生成 39_vae_gan_flow.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 39 — VAE、GAN、Normalizing Flow：经典生成模型

> 扩散之前，三大生成模型各领风骚。
> 🪡 GAN 用博弈论，VAE 用变分推断，Flow 用可逆变换。

## 本章你将掌握

1. **VAE**：变分下界 + 重参数化
2. **GAN**：🪡 纳什均衡 + 对抗训练
3. **Normalizing Flow**：可逆变换 + Jacobian
4. **三大模型对比**""")

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

md("""## 1. VAE：变分自编码器

### 1.1 问题

普通自编码器：编码 → 潜空间 → 解码，但潜空间**不连续**——不能随机采样生成。

### 1.2 VAE 的解决方案

让潜空间**接近先验** $\\mathcal{N}(0, I)$：

```
编码器: x → (μ, σ) → z = μ + σ * ε  (重参数化)
解码器: z → x̂

损失 (ELBO):
  重建项: E[log p(x|z)]  → 重建好
  正则项: KL(q(z|x) || N(0,I))  → 潜空间规整
```

### 1.3 重参数化技巧

```
不能直接采样 z ~ N(μ, σ)
→ z = μ + σ * ε, ε ~ N(0, 1)
→ 梯度可以通过 μ, σ 回传
```

> 💡 VAE 的核心：**让潜空间规整**——重建和正则的权衡。
# KL 散度把潜空间拉向标准正态，使随机采样有意义。""")

code("""# VAE 实现
class VAE(nn.Module):
    def __init__(self, data_dim=784, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def loss(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        # 重建损失
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        # KL 散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl_loss) / x.shape[0]

# 训练 VAE
vae = VAE(data_dim=78, latent_dim=16, hidden_dim=64)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# 模拟数据
data = torch.randn(500, 78) * 0.5 + 0.5
losses = []
for _ in range(1000):
    loss = vae.loss(data)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())

# 生成
with torch.no_grad():
    z = torch.randn(100, 16)
    samples = vae.decode(z)

print(f"VAE 训练: loss {losses[0]:.2f} → {losses[-1]:.2f}")
print(f"潜空间维度: 16, 生成样本: {samples.shape}")
print("VAE: 重建 + KL 正则 → 规整潜空间 → 可采样生成。")""")

md("""## 2. GAN：生成对抗网络

### 2.1 🪡 博弈论视角

GAN 是一个**二人零和博弈**：

```
生成器 G: 噪声 → 假数据 (骗过判别器)
判别器 D: 数据 → 真/假 (区分真假)

博弈:
  G 想: max D(假) → 让 D 把假的判为真
  D 想: max D(真) - D(假) → 区分真假
```

### 2.2 纳什均衡

```
纯策略纳什均衡:
  G 生成分布 = 真实分布
  D 输出 = 0.5 (无法区分)

此时: 任何一方单方面改变策略都不会更好
→ 纳什均衡
```

### 2.3 训练不稳定

```
问题: 纳什均衡是鞍点, 不是极小值
  → 梯度下降在鞍点附近震荡
  → 模式坍缩: G 只生成几种样本

解决:
  - WGAN: 用 Wasserstein 距离 (更稳定)
  - 谱归一化: 约束 D 的 Lipschitz 常数
  - 渐进式: 从低分辨率逐步增长
```

> 💡 GAN 的博弈论本质：**生成器和判别器的纳什均衡**。
# 但鞍点让训练困难——这是 GAN 被扩散取代的主要原因。""")

code("""# GAN 实现
class Generator(nn.Module):
    def __init__(self, noise_dim=32, data_dim=78, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim=78, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

G = Generator(noise_dim=32, data_dim=78, hidden_dim=128)
D = Discriminator(data_dim=78, hidden_dim=128)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# 训练 GAN
real_data = torch.randn(500, 78) * 0.5 + 0.5
d_losses, g_losses = [], []

for epoch in range(2000):
    # 训练 D
    real = real_data[torch.randint(0, 500, (64,))]
    z = torch.randn(64, 32)
    fake = G(z).detach()

    d_real = D(real)
    d_fake = D(fake)
    d_loss = -(torch.log(torch.sigmoid(d_real) + 1e-8).mean() +
               torch.log(1 - torch.sigmoid(d_fake) + 1e-8).mean())

    opt_D.zero_grad(); d_loss.backward(); opt_D.step()

    # 训练 G
    z = torch.randn(64, 32)
    fake = G(z)
    g_loss = -torch.log(torch.sigmoid(D(fake)) + 1e-8).mean()

    opt_G.zero_grad(); g_loss.backward(); opt_G.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(d_losses, 'r-', linewidth=1, label='D loss')
ax.plot(g_losses, 'b-', linewidth=1, label='G loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('GAN 对抗训练'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_gan.png', bbox_inches='tight')
plt.show()
print(f"GAN: D loss {d_losses[-1]:.3f}, G loss {g_losses[-1]:.3f}")
print("GAN: 生成器和判别器博弈——纳什均衡时 D=0.5。")""")

md("""## 3. Normalizing Flow：可逆变换

### 3.1 核心思想

通过一系列**可逆变换**把简单分布变成复杂分布：

```
z0 ~ N(0, I)  (简单)
z1 = f1(z0)   (可逆变换)
z2 = f2(z1)
...
x = zn = fn(...f1(z0))  (复杂)

概率密度变换:
  log p(x) = log p(z0) - Σ log |det dfi/dzi|
```

### 3.2 精确对数似然

Flow 的优势：**精确的对数似然**（VAE 只有下界）。

```
log p(x) = log p(z0) - Σ log |det Jacobian|

→ 可以精确优化对数似然
→ 不像 VAE 有 gap
```

### 3.3 RealNVP

常用的 Flow 架构：

```
把 x 分成 (x1, x2)
  x1' = x1
  x2' = x2 * exp(s(x1)) + t(x1)

Jacobian 是三角矩阵 → det 容易计算
```

> 💡 Flow 的优势：精确似然，可逆生成。但变换受限（必须可逆），
# 表达力不如扩散——这是它没成为主流的原因。""")

code("""# Normalizing Flow (RealNVP) 简化
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.half = dim // 2
        self.s_net = nn.Sequential(nn.Linear(self.half, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, self.half))
        self.t_net = nn.Sequential(nn.Linear(self.half, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, self.half))

    def forward(self, x):
        x1, x2 = x[:, :self.half], x[:, self.half:]
        s = self.s_net(x1)
        t = self.t_net(x1)
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=-1)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y):
        y1, y2 = y[:, :self.half], y[:, self.half:]
        s = self.s_net(y1)
        t = self.t_net(y1)
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=-1)
        return x

class Flow(nn.Module):
    def __init__(self, dim=4, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(dim) for _ in range(n_layers)])

    def forward(self, z):
        log_det_total = 0
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x):
        # 逆变换: x → z, 计算 log p(x)
        z = x
        log_det_total = 0
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        # log p(z) + log |det dz/dx|
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * z.shape[-1] * np.log(2 * np.pi)
        # 正向 log_det
        _, log_det = self.forward(z)
        return log_pz + log_det

# 训练 Flow
flow = Flow(dim=4, n_layers=4)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

# 目标: 二模分布
def bimodal(n=256):
    mode = np.random.choice(2, n)
    data = np.where(mode[:, None], np.random.randn(n, 4) + 3, np.random.randn(n, 4) - 3)
    return torch.tensor(data, dtype=torch.float32)

losses = []
for _ in range(2000):
    data = bimodal(256)
    log_prob = flow.log_prob(data)
    loss = -log_prob.mean()  # 负对数似然
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())

# 生成
with torch.no_grad():
    z = torch.randn(500, 4)
    samples, _ = flow(z)

print(f"Flow 训练: loss {losses[0]:.2f} → {losses[-1]:.2f}")
print(f"生成样本: {samples.shape}")
print("Flow: 精确对数似然, 可逆变换——但表达力受限。")""")

md("""## 4. 三大模型 + 扩散 对比

| 模型 | 似然 | 采样 | 训练 | 质量 |
|------|------|------|------|------|
| **VAE** | 下界 (ELBO) | 快 (1步) | 稳定 | 模糊 |
| **GAN** | 无 | 快 (1步) | 不稳定 | 清晰 |
| **Flow** | 精确 | 快 (1步) | 稳定 | 中等 |
| **扩散** | 精确 | 慢 (多步) | 稳定 | 最好 |

### 为什么扩散赢了

```
VAE:  模糊 (ELBO 有 gap)
GAN:  训练不稳定 (鞍点), 模式坍缩
Flow: 变换受限, 表达力不够
扩散: 训练稳定 + 质量最好 (但慢)
```

> 💡 扩散模型综合了 VAE 的稳定性和 GAN 的质量——
# 虽然慢，但质量和稳定性完胜。加速方法（DDIM/Flow Matching）解决了速度问题。

### 🔗 下一章
**`40_autoregressive_vs_diffusion.ipynb`** — 自回归 vs 扩散对比

---

> 💬 **板块六进行中。**""")

output_path = "notebooks/39_vae_gan_flow.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")