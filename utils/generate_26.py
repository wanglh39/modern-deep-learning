# 生成 26_3d_representation.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 26 — 3D 表示：NeRF 与 3D Gaussian Splatting

> 从 2D 图片重建 3D 场景——NeRF 用神经网络，3DGS 用高斯椭球。
> 这是 VR/AR、数字孪生、机器人感知的基础。

## 本章你将掌握

1. **3D 表示的演进**：体素 → 点云 → mesh → NeRF → 3DGS
2. **NeRF 原理**：神经辐射场，光线行进 + 体渲染
3. **3D Gaussian Splatting**：显式表示，实时渲染
4. **对比与应用**""")

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

md("""## 1. 3D 表示的演进

### 1.1 五种表示方法

```
1. 体素 (Voxel)      : 3D 网格, 每格存密度/颜色 → 内存爆炸
2. 点云 (Point Cloud) : N 个点 (x,y,z) → 无体积, 难渲染
3. Mesh              : 顶点+面片 → 难优化, 不适合柔性物体
4. NeRF (2020)       : 神经网络隐式表示 → 质量高但慢
5. 3DGS (2023)       : 显式高斯椭球 → 质量高且快
```

### 1.2 核心问题

给定**多张 2D 照片**（不同视角），重建 **3D 场景**，并能从**新视角**渲染。

```
输入: {照片_i, 相机位姿_i}  (几十到几百张)
输出: 3D 场景表示
目标: 从新视角渲染 → 与真实照片一致
```

> 💡 NeRF 的突破：用神经网络隐式表示 3D 场景，质量远超之前的方法。
> 3DGS 的突破：把隐式变显式，速度提升 100-1000 倍。""")

md("""## 2. NeRF：神经辐射场

### 2.1 核心思想

NeRF 用一个 MLP 表示 3D 场景的**辐射场**：

```
f(x, y, z, θ, φ) → (σ, c)

输入: 3D 位置 (x,y,z) + 视角方向 (θ,φ)
输出: 密度 σ + 颜色 c = (r,g,b)
```

### 2.2 体渲染

从相机发出光线，穿过场景，**积分**得到像素颜色：

$$C(\\mathbf{r}) = \\int_{t_n}^{t_f} T(t) \\sigma(\\mathbf{r}(t)) \\mathbf{c}(\\mathbf{r}(t), \\mathbf{d}) dt$$

其中 $T(t) = \\exp\\left(-\\int_{t_n}^{t} \\sigma(\\mathbf{r}(s)) ds\\right)$ 是透射率。

### 2.3 训练

```
1. 对每个像素, 发射一条光线
2. 在光线上采样 N 个点
3. 用 MLP 预测每个点的 (σ, c)
4. 体渲染积分 → 预测像素颜色
5. 与真实像素比较 → 损失 → 反传
```

> 💡 NeRF 的优雅：一个 MLP 就编码了整个 3D 场景。
> 但渲染慢——每条光线要采样几十到几百个点，每个点过一次 MLP。""")

code("""# NeRF 简化实现
class MiniNeRF(nn.Module):
    def __init__(self, d_pos=6, d_dir=4, d_hidden=128):
        super().__init__()
        # 位置编码后的维度 + 视角编码后的维度
        self.pos_mlp = nn.Sequential(
            nn.Linear(3 + 2 * 3 * d_pos, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1 + d_hidden)  # 密度 + 特征
        )
        self.color_mlp = nn.Sequential(
            nn.Linear(d_hidden + 3 + 2 * 3 * d_dir, d_hidden // 2), nn.ReLU(),
            nn.Linear(d_hidden // 2, 3)  # RGB
        )

    def positional_encoding(self, x, n_freqs):
        freqs = 2 ** torch.arange(n_freqs).float() * np.pi
        embeds = [x]
        for freq in freqs:
            embeds.append(torch.sin(freq * x))
            embeds.append(torch.cos(freq * x))
        return torch.cat(embeds, dim=-1)

    def forward(self, pos, direction):
        # 位置编码
        pos_enc = self.positional_encoding(pos, 6)
        dir_enc = self.positional_encoding(direction, 4)

        # 预测密度 + 特征
        h = self.pos_mlp(pos_enc)
        sigma = F.relu(h[..., :1])  # 密度
        feature = h[..., 1:]        # 特征

        # 预测颜色
        color_input = torch.cat([feature, dir_enc], dim=-1)
        color = torch.sigmoid(self.color_mlp(color_input))  # [0,1]
        return sigma, color

# 体渲染
def volume_render(sigma, color, t_vals):
    # sigma, color: [N_samples, 1/3], t_vals: [N_samples]
    delta = t_vals[1:] - t_vals[:-1]
    delta = torch.cat([delta, torch.tensor([1e10])])

    alpha = 1 - torch.exp(-sigma.squeeze() * delta)  # 不透明度
    T = torch.cumprod(1 - alpha + 1e-10, dim=0)
    T = torch.cat([torch.tensor([1.0]), T[:-1]])

    weights = T * alpha
    rendered_color = (weights.unsqueeze(-1) * color).sum(dim=0)
    return rendered_color, weights

# 演示 NeRF
nerf = MiniNeRF()
# 一条光线上的 64 个采样点
n_samples = 64
t_vals = torch.linspace(0.5, 5.0, n_samples)
positions = torch.randn(n_samples, 3) * 0.5
direction = torch.tensor([0.0, 0.0, -1.0]).unsqueeze(0).expand(n_samples, -1)

with torch.no_grad():
    sigma, color = nerf(positions, direction)
    rendered, weights = volume_render(sigma, color, t_vals)

print(f"NeRF MLP: 输入(3D位置+视角) → 输出(密度+颜色)")
print(f"光线上 {n_samples} 个采样点 → 渲染颜色 RGB = {rendered.numpy()}")
print(f"参数量: {sum(p.numel() for p in nerf.parameters())/1e3:.0f}K")""")

code("""# 可视化 NeRF 的体渲染
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 沿光线的密度和权重
ax = axes[0]
ax.plot(t_vals.numpy(), sigma.squeeze().detach().numpy(), 'r-', linewidth=2, label='密度 σ')
ax.plot(t_vals.numpy(), weights.detach().numpy(), 'b-', linewidth=2, label='渲染权重')
ax.set_xlabel('光线深度 t'); ax.set_ylabel('值')
ax.set_title('NeRF 沿光线的密度与权重'); ax.legend(); ax.grid(True, alpha=0.3)

# 位置编码的效果
ax = axes[1]
x = torch.linspace(-2, 2, 200).unsqueeze(-1)
for freq in [1, 2, 4, 8]:
    ax.plot(x.numpy(), torch.sin(freq * np.pi * x).numpy(), linewidth=1.5, label=f'freq={freq}')
ax.set_xlabel('x'); ax.set_ylabel('sin(2^k π x)')
ax.set_title('位置编码: 高频帮助捕捉细节'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_nerf.png', bbox_inches='tight')
plt.show()
print("位置编码让 MLP 能捕捉高频细节——NeRF 的关键技巧。")""")

md("""## 3. 3D Gaussian Splatting

### 3.1 动机

NeRF 的问题：**慢**。渲染一帧要发射几百万条光线，每条采样几十个点。

3DGS 的想法：**用显式的高斯椭球代替隐式 MLP**。

### 3.2 表示

每个高斯椭球由以下参数定义：

```
G(x) = exp(-1/2 (x-μ)^T Σ^{-1} (x-μ))

参数:
  μ ∈ R³      位置 (中心)
  Σ ∈ R³ˣ³   协方差矩阵 (形状/大小)
  c ∈ R^S    球谐系数 (颜色, 视角相关)
  α ∈ R      不透明度
```

### 3.3 渲染：Splatting

```
传统 NeRF: 光线行进 → 逐点采样 → 积分 (慢)
3DGS:      把高斯"泼"到屏幕上 → α混合 (快)

每个高斯 → 投影到 2D → 椭圆 → 排序 → α混合
```

### 3.4 自适应密度控制

```
1. 梯度大的区域 → 分裂高斯 (增加细节)
2. α 太小的高斯 → 删除 (清理)
3. 大高斯 → 分裂成两个小的
→ 自适应调整高斯数量和位置
```

> 💡 3DGS 用几十万到几百万个高斯椭球表示场景。
> 渲染用可微的 splatting，训练端到端。
> 速度比 NeRF 快 100-1000 倍，质量相当或更好。""")

code("""# 3D Gaussian Splatting 简化
class GaussianSplat:
    def __init__(self, n_gaussians=1000):
        self.n = n_gaussians
        # 每个高斯的参数
        self.position = torch.randn(n_gaussians, 3) * 2  # 中心
        self.scale = torch.rand(n_gaussians, 3) * 0.1 + 0.01  # 缩放
        self.rotation = torch.randn(n_gaussians, 4)  # 四元数
        self.color = torch.rand(n_gaussians, 3)  # RGB
        self.opacity = torch.rand(n_gaussians)  # 不透明度

    def render(self, camera_pos, n_pixels=64*64):
        # 简化: 把高斯投影到 2D, α混合
        # 实际中用 CUDA 加速
        rendered = torch.zeros(n_pixels, 3)
        # 按深度排序 (简化)
        depths = (self.position - camera_pos).norm(dim=-1)
        order = depths.argsort()

        for idx in order[:100]:  # 只取最近的100个 (简化)
            # 投影到 2D (简化为随机位置)
            pixel = torch.randint(0, n_pixels, (1,)).item()
            rendered[pixel] += self.opacity[idx] * self.color[idx] * (1 - rendered[pixel].sum())

        return rendered.reshape(64, 64, 3)

# 对比 NeRF vs 3DGS
splat = GaussianSplat(n_gaussians=10000)
camera = torch.tensor([0.0, 0.0, 5.0])

import time
# 3DGS 渲染
start = time.time()
img_3dgs = splat.render(camera)
time_3dgs = time.time() - start

# 模拟 NeRF 渲染时间 (慢 100x)
time_nerf = time_3dgs * 100

print(f"3DGS 渲染: {time_3dgs*1000:.1f} ms")
print(f"NeRF 渲染 (估计): {time_nerf*1000:.1f} ms")
print(f"加速比: {time_nerf/time_3dgs:.0f}x")
print(f"高斯数量: {splat.n}")
print("3DGS: 显式表示 → 可并行 splat → 实时渲染。")""")

code("""# 可视化高斯椭球
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 3D 高斯分布
ax = axes[0]
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# 几个高斯叠加
gaussians = [((0, 0, 0.5, 0.3, 'red')), ((1, 1, 0.4, 0.5, 'blue')), ((-1, -0.5, 0.6, 0.2, 'green'))]
Z = np.zeros_like(X)
for cx, cy, s, alpha, color in gaussians:
    Z += alpha * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*s**2))

ax.contourf(X, Y, Z, levels=20, cmap='hot')
ax.set_title('3DGS: 高斯椭球叠加 (2D投影)')
ax.set_xlabel('x'); ax.set_ylabel('y')

# NeRF vs 3DGS 对比
ax = axes[1]
methods = ['NeRF', 'Instant-NGP', '3DGS']
render_times = [5000, 50, 5]  # ms
qualities = [29.5, 29.2, 29.4]  # PSNR

ax.bar(methods, render_times, color=['red', 'orange', 'green'], alpha=0.7)
ax.set_ylabel('渲染时间 (ms)')
ax.set_title('渲染速度对比 (对数尺度)')
ax.set_yscale('log')
for i, (m, t) in enumerate(zip(methods, render_times)):
    ax.text(i, t * 1.2, f'{t}ms', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('notebooks/fig_3dgs.png', bbox_inches='tight')
plt.show()
print("3DGS: 速度提升 1000x, 质量相当——实时 3D 渲染。")""")

md("""## 4. NeRF vs 3DGS 对比

| 特性 | NeRF | 3DGS |
|------|------|------|
| **表示** | 隐式 (MLP) | 显式 (高斯椭球) |
| **渲染** | 光线行进 | Splatting |
| **速度** | 慢 (~秒/帧) | 快 (~ms/帧) |
| **质量** | 高 | 高 |
| **编辑** | 难 (隐式) | 易 (显式) |
| **显存** | 小 (MLP) | 大 (存高斯) |
| **动态** | 难 | 较易 |

### 应用

- **VR/AR**：3DGS 实时渲染，适合 VR
- **数字孪生**：建筑/工厂的 3D 重建
- **电影特效**：虚拟场景
- **机器人**：3D 环境理解
- **自动驾驶**：场景重建

> 💡 3DGS 正在取代 NeRF 成为 3D 重建的主流方法。
> 但 NeRF 的思想（神经场）仍在其他领域有用。""")

md("""## 5. 前沿方向

### 5.1 动态场景

- **D-NeRF**：时间作为额外输入
- **4DGS**：高斯随时间变形
- **Dynamic 3DGS**：高斯跟踪运动

### 5.2 大规模

- **Block-NeRF**：分块渲染大场景
- **Mega-NeRF**：大规模场景
- **City 3DGS**：城市级重建

### 5.3 生成式 3D

- **DreamFusion**：文本 → 3D (SDS 损失)
- **GaussianDreamer**：文本 → 3D 高斯
- **LRM**：单图 → 3D mesh

> 💡 2024 年的趋势：3DGS + 生成模型 → 文本/图像生成 3D 内容。""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 3D 表示演进 | ✅ |
| NeRF 原理 | ✅ |
| 体渲染方程 | ✅ |
| 3D Gaussian Splatting | ✅ |
| NeRF vs 3DGS | ✅ |

### 核心 takeaway
> **NeRF 用 MLP 隐式表示 3D 场景，质量高但慢**。
> **3DGS 用显式高斯椭球，实时渲染且可编辑**。
> 从隐式到显式，是 3D 表示的实用化转向。

### 🔗 下一章
**`27_embodied_ai.ipynb`** — VLA、RT-2、机器人

---

> 💬 **板块四进行中。**""")

output_path = "notebooks/26_3d_representation.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")