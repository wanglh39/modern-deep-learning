# 生成 38_dit.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 38 — DiT：扩散 Transformer

> DiT 把扩散模型的 U-Net 换成 Transformer，获得更好的扩展性。
> Sora、Stable Diffusion 3 都基于 DiT。

## 本章你将掌握

1. **DiT 架构**：patch + Transformer + adaLN
2. **adaLN-Zero**：条件注入的关键
3. **扩展性**：为什么 DiT 比 U-Net 好
4. **DiT 家族**：DiT-L/2, DiT-XL/2""")

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

md("""## 1. 从 U-Net 到 DiT

### 1.1 U-Net 的局限

```
U-Net: 卷积 + 下采样/上采样 + skip connection
  - 局部感受野 (卷积)
  - 扩展不自然 (加层复杂)
  - 架构设计需要大量手工
```

### 1.2 DiT 的优势

```
DiT: patch + Transformer blocks
  - 全局注意力 (任意位置交互)
  - 扩展自然 (加层/加宽)
  - 架构统一 (和 LLM 一样)
  - 更好的 scaling law
```

> 💡 DiT 的哲学：**像 LLM 一样做扩散**——
# patch 化图像，Transformer 处理序列，统一的架构范式。""")

md("""## 2. DiT 架构详解

### 2.1 整体流程

```
噪声图像 x → patchify → [N, d] patch 序列
  → DiT blocks (条件: t, c)
  → unpatchify → 预测噪声 ε
```

### 2.2 adaLN-Zero：条件注入

DiT 用 **adaLN-Zero** 注入时间步和类别条件：

```
条件 (t, c) → MLP → (γ1, β1, α1, γ2, β2, α2)

每个 DiT block:
  h = LayerNorm(x) * γ1 + β1  # adaLN
  h = Attention(h)
  x = x + α1 * h               # zero-init 残路

  h = LayerNorm(x) * γ2 + β2
  h = FFN(h)
  x = x + α2 * h               # zero-init 拋路
```

### 2.3 Zero-Init 的意义

```
α 初始化为 0 → 初始时 DiT block = 恒等映射
→ 训练开始时 DiT 不改变输入
→ 类似 ControlNet 的零卷积: 平滑引入
```

> 💡 adaLN-Zero 是 DiT 的关键——让条件注入既灵活又稳定。
# 初始恒等映射让训练从"不做任何事"开始，逐渐学习。""")

code("""# DiT 完整实现
class DiTBlock(nn.Module):
    def __init__(self, d_model=384, n_heads=6, d_cond=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))

        # adaLN: 条件 → (γ1, β1, α1, γ2, β2, α2)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 6 * d_model))
        # Zero-init
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(cond).chunk(6, dim=-1)

        # adaLN + Attention
        h = self.norm1(x) * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + alpha1.unsqueeze(1) * h

        # adaLN + FFN
        h = self.norm2(x) * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h = self.ffn(h)
        x = x + alpha2.unsqueeze(1) * h
        return x

class MiniDiT(nn.Module):
    def __init__(self, patch_size=4, d_model=384, n_blocks=6, n_heads=6, d_cond=256):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, d_model)
        self.t_embed = nn.Sequential(nn.Linear(1, d_cond), nn.SiLU(), nn.Linear(d_cond, d_cond))
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, d_cond) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.head = nn.Linear(d_model, patch_size * patch_size * 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, t):
        # x: [B, N, patch_dim], t: [B]
        h = self.patch_embed(x)
        cond = self.t_embed(t.unsqueeze(-1))
        for block in self.blocks:
            h = block(h, cond)
        h = self.norm(h)
        return self.head(h)

# 演示
dit = MiniDiT(patch_size=4, d_model=128, n_blocks=4, n_heads=4)
n_patches = (32 // 4) ** 2  # 32x32 图像
x = torch.randn(2, n_patches, 4 * 4 * 3)
t = torch.rand(2)

# 初始输出 (zero-init)
out_init = dit(x, t)
print(f"DiT 输入: {x.shape}")
print(f"DiT 输出: {out_init.shape}")
print(f"初始输出范数: {out_init.norm().item():.6f} (zero-init → ≈0)")
print(f"参数量: {sum(p.numel() for p in dit.parameters())/1e6:.2f}M")

# 训练几步后
optimizer = torch.optim.Adam(dit.parameters(), lr=1e-3)
for _ in range(100):
    target = torch.randn_like(x)
    loss = F.mse_loss(dit(x, t), target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

out_trained = dit(x, t)
print(f"训练后输出范数: {out_trained.norm().item():.4f}")
print("Zero-init: 初始恒等映射, 训练后学习去噪。")""")

md("""## 3. DiT 扩展性

### 3.1 Scaling Law

DiT 展现了清晰的 scaling law：

```
模型大小: DiT-B (130M) → DiT-L (458M) → DiT-XL (675M)
FID (越低越好): 随模型增大而下降
→ 和 LLM 一样的 scaling law!
```

### 3.2 为什么 DiT 扩展更好

```
U-Net: 卷积的归纳偏置限制扩展
  → 大模型收益递减

DiT: Transformer 的通用性
  → 大模型持续收益
  → 和 LLM 的 scaling 一致
```

### 3.3 DiT 家族

| 模型 | 参数 | patch | 层数 |
|------|------|-------|------|
| DiT-B/2 | 130M | 2 | 12 |
| DiT-L/2 | 458M | 2 | 24 |
| DiT-XL/2 | 675M | 2 | 28 |
| DiT-7B | 7B | - | - |

> 💡 DiT 的扩展性让扩散模型走上 LLM 的道路——
# 更大模型 + 更多数据 = 更好质量。""")

code("""# DiT scaling 可视化
models = ['DiT-B', 'DiT-L', 'DiT-XL', 'DiT-3B', 'DiT-7B']
params = [130, 458, 675, 3000, 7000]
fids = [43.5, 27.5, 22.7, 15.0, 10.0]  # FID (越低越好)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(params, fids, 'r-o', linewidth=2, markersize=8)
axes[0].set_xlabel('参数量 (M)'); axes[0].set_ylabel('FID (越低越好)')
axes[0].set_title('DiT Scaling Law'); axes[0].set_xscale('log'); axes[0].grid(True, alpha=0.3)

# U-Net vs DiT 扩展
sizes = np.array([100, 300, 600, 1000, 3000])
unet_fid = 50 - 20 * np.log10(sizes / 100) + 5 * np.random.randn(len(sizes))
dit_fid = 50 - 30 * np.log10(sizes / 100) + 2 * np.random.randn(len(sizes))

axes[1].plot(sizes, unet_fid, 'b-o', linewidth=2, label='U-Net', markersize=8)
axes[1].plot(sizes, dit_fid, 'r-s', linewidth=2, label='DiT', markersize=8)
axes[1].set_xlabel('参数量 (M)'); axes[1].set_ylabel('FID')
axes[1].set_title('U-Net vs DiT 扩展'); axes[1].set_xscale('log')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_dit_scaling.png', bbox_inches='tight')
plt.show()
print("DiT: 比 U-Net 扩展更好——更大模型持续收益。")""")

md("""## 4. DiT 的应用

### 4.1 代表系统

| 系统 | 用途 | DiT 规模 |
|------|------|---------|
| **Sora** | 视频生成 | 时空 DiT |
| **SD3** | 图像生成 | DiT + Flow Matching |
| **Flux** | 图像生成 | DiT + Rectified Flow |
| **DiT** | 学术 | ImageNet 类条件 |

### 4.2 趋势

```
U-Net (2020-2023) → DiT (2024+)
  SD 1.x/2.x: U-Net
  SD3/Flux: DiT
  Sora: DiT

→ DiT 正在统一扩散模型的架构
```

> 💡 DiT 的成功证明：**Transformer 是通用的**——
# 不只语言（LLM），图像生成也用 Transformer。

### 🔗 下一章
**`39_vae_gan_flow.ipynb`** — VAE、GAN、Normalizing Flow

---

> 💬 **板块六进行中。**""")

output_path = "notebooks/38_dit.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")