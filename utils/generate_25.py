# 生成 25_world_models_video.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 25 — 世界模型与视频生成：Sora

> Sora 不是简单的视频生成器，它是一个**世界模型**——理解物理规律、物体持久性、3D 空间。
> 这一章从 DiT 到时空 patch，解析世界模型的架构。

## 本章你将掌握

1. **视频生成的挑战**：时间一致性、长视频
2. **DiT 架构**：Diffusion + Transformer
3. **时空 patch**：Sora 的核心表示
4. **世界模型**：不只是生成，而是理解""")

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

md("""## 1. 视频生成的挑战

### 1.1 图像 vs 视频

```
图像: H × W × 3              (一帧)
视频: T × H × W × 3          (T 帧, 每帧 H×W×3)

参数量爆炸:
  1帧 256×256×3 = 196,608
  60帧 256×256×3 = 11,796,480  (60倍!)
```

### 1.2 三大挑战

| 挑战 | 说明 |
|------|------|
| **时间一致性** | 帧间物体不能突然消失/变形 |
| **长视频** | 60秒视频 = 1500帧，如何保持一致 |
| **物理规律** | 物体运动要符合物理（重力、碰撞） |

> 💡 早期视频生成用 3D UNet，但难以扩展到长视频。
> Sora 用 Transformer + 时空 patch 解决了这些问题。""")

md("""## 2. DiT：Diffusion Transformer

### 2.1 从 UNet 到 Transformer

```
传统扩散模型 (DDPM, Stable Diffusion):
  去噪网络 = UNet (卷积 + 下采样/上采样)

DiT (Diffusion Transformer):
  去噪网络 = Transformer (patch + attention)
```

### 2.2 DiT 架构

```
噪声图像 → patchify → [N patches, d]
                            ↓
                    Transformer blocks
                    (每层: self-attn + MLP)
                            ↓
                    unpatchify → 预测噪声
```

### 2.3 为什么 DiT 更好

- **可扩展性**：Transformer 加层/加宽更自然
- **全局注意力**：不像卷积有局部感受野限制
- **训练效率**：更大的模型 + 更多的数据

> 💡 Sora 基于 DiT——把图像扩散的 UNet 换成 Transformer，获得扩展性。
> Stable Diffusion 3 也转向 DiT 架构。""")

code("""# DiT 简化实现
class DiTBlock(nn.Module):
    def __init__(self, d_model=384, n_heads=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # 自注意力
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x

class MiniDiT(nn.Module):
    def __init__(self, patch_size=16, d_model=384, n_blocks=6, n_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, d_model)
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, patch_size * patch_size * 3)

    def forward(self, x):
        # x: [batch, n_patches, patch_size^2 * 3]
        h = self.patch_embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h)

# 演示 DiT
dit = MiniDiT(patch_size=16, d_model=384, n_blocks=6)
n_patches = (256 // 16) ** 2  # 256x256 图像, 16x16 patch → 256 patches
x = torch.randn(2, n_patches, 16 * 16 * 3)
out = dit(x)
print(f"输入: {x.shape} (2张图, {n_patches}个patch)")
print(f"输出: {out.shape} (预测每个patch的噪声)")
n_params = sum(p.numel() for p in dit.parameters())
print(f"参数量: {n_params/1e6:.1f}M")
print("DiT: Transformer 做去噪——可扩展、全局注意力。")""")

md("""## 3. 时空 patch：Sora 的核心

### 3.1 把视频变成 patch

Sora 的关键创新：把视频切成 **时空 patch (spatiotemporal patch)**：

```
视频: T × H × W × 3
  → 切成 (t × h × w) 的 3D patch
  → 每个 patch 是一小段时空立方体
  → 展平为 token 序列
```

### 3.2 举例

```
视频: 60帧 × 256×256 × 3
patch 大小: 2帧 × 16×16 × 3
→ (60/2) × (256/16) × (256/16) = 30 × 16 × 16 = 7680 个 patch
→ 每个 patch 是 2帧×16×16×3 = 1536 维
```

### 3.3 优势

- **统一处理**：不同分辨率/帧率只需调整 patch 数量
- **时空局部性**：每个 patch 包含局部时空信息
- **Transformer 友好**：天然适合序列建模

> 💡 时空 patch 让 Sora 能处理**任意分辨率、任意时长**的视频——
> 训练时不需要统一尺寸，大幅简化数据管道。""")

code("""# 时空 patch 演示
def video_to_spatiotemporal_patches(video, patch_t=2, patch_h=16, patch_w=16):
    # video: [T, H, W, 3]
    T, H, W, C = video.shape
    n_t = T // patch_t
    n_h = H // patch_h
    n_w = W // patch_w

    # 切成 patch
    patches = video[:n_t*patch_t, :n_h*patch_h, :n_w*patch_w, :]
    patches = patches.reshape(n_t, patch_t, n_h, patch_h, n_w, patch_w, C)
    patches = patches.transpose(0, 2, 4, 1, 3, 5, 6)  # [n_t, n_h, n_w, patch_t, patch_h, patch_w, C]
    patches = patches.reshape(-1, patch_t * patch_h * patch_w * C)  # [N_patches, patch_dim]

    return patches, (n_t, n_h, n_w)

# 模拟视频
video = np.random.randn(60, 256, 256, 3).astype(np.float32)  # 60帧 256x256
patches, (n_t, n_h, n_w) = video_to_spatiotemporal_patches(video, patch_t=2, patch_h=16, patch_w=16)

print(f"视频: {video.shape[0]}帧 × {video.shape[1]}×{video.shape[2]} × {video.shape[3]}通道")
print(f"时空 patch: 2帧×16×16×3 = {2*16*16*3}维")
print(f"patch 数量: {n_t}×{n_h}×{n_w} = {n_t*n_h*n_w}")
print(f"patch 矩阵: {patches.shape}")
print(f"\\n不同视频尺寸 → 不同 patch 数量, 但 patch 维度不变:")
for T, H, W in [(30, 128, 128), (60, 256, 256), (120, 512, 512)]:
    v = np.random.randn(T, H, W, 3).astype(np.float32)
    p, (nt, nh, nw) = video_to_spatiotemporal_patches(v)
    print(f"  {T}帧×{H}×{W} → {nt}×{nh}×{nw} = {nt*nh*nw} patches")""")

md("""## 4. Sora 的架构

### 4.1 整体流程

```
文本 prompt → 文本编码器 → 文本嵌入
                                ↓
噪声视频 → patchify → 时空 patch → DiT 去噪 (条件: 文本嵌入)
                                        ↓
                                unpatchify → 生成视频
```

### 4.2 关键组件

| 组件 | 作用 |
|------|------|
| **时空 VAE** | 压缩视频到潜空间 (时间+空间压缩) |
| **DiT** | 在潜空间去噪 |
| **文本编码器** | 条件控制 |
| **patchify** | 统一处理不同尺寸 |

### 4.3 训练数据

Sora 训练数据规模巨大：
- 视频时长：几秒到几分钟
- 分辨率：各种
- DALL·E 3 重写 prompt（提升文本质量）

> 💡 Sora 的核心不是某个算法创新，而是**工程系统**：
> DiT + 时空 patch + 大规模数据 + 大模型。""")

code("""# Sora 生成过程模拟 (简化)
def sora_generation_sim(prompt, n_frames=60, h=256, w=256, n_steps=10):
    # 模拟时空 VAE 压缩
    latent_t, latent_h, latent_w = n_frames // 4, h // 16, w // 16
    print(f"输入: '{prompt}'")
    print(f"目标视频: {n_frames}帧 × {h}×{w}")
    print(f"潜空间: {latent_t}×{latent_h}×{latent_w} (压缩 {n_frames*h*w/(latent_t*latent_h*latent_w):.0f}x)")

    # 模拟扩散去噪
    latent = torch.randn(1, latent_t * latent_h * latent_w, 384)
    losses = []
    for step in range(n_steps):
        # 模拟去噪: loss 下降
        loss = 2.0 * np.exp(-step / 3) + 0.1
        losses.append(loss)

    return losses

losses = sora_generation_sim("一只猫在月光下的屋顶上走路", n_frames=60)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(losses, 'b-o', linewidth=2, markersize=6)
ax.set_xlabel('去噪步数'); ax.set_ylabel('噪声水平')
ax.set_title('Sora 扩散去噪过程'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_sora_denoise.png', bbox_inches='tight')
plt.show()
print("从纯噪声逐步去噪 → 生成视频。")""")

md("""## 5. 世界模型：不只是生成

### 5.1 Sora 展现的"涌现"能力

Sora 不只是模仿训练数据，它展现了**理解物理世界**的能力：

```
✅ 物体持久性: 物体被遮挡后重新出现, 保持一致
✅ 3D 一致性: 相机移动时, 3D 结构一致
✅ 重力: 物体下落符合重力
✅ 反射: 镜面/水面反射正确
❌ 精确物理: 碰撞、流体仍有错误
❌ 因果: 因果关系有时混乱
```

### 5.2 世界模型 vs 生成模型

```
生成模型: 学习数据分布 p(x), 采样生成
世界模型: 学习世界动力学, 能预测/规划

区别: 世界模型理解"为什么", 生成模型只学"什么样"
```

### 5.3 LeCun 的世界模型愿景

Yann LeCun 提出的 JEPA（Joint Embedding Predictive Architecture）：

```
观察 x → 编码 → 表示 s
行动 a → 预测下一个表示 s'
        ↓
不预测像素, 预测抽象表示 → 免受像素噪声干扰
```

> 💡 争论：Sora 是不是"真正的"世界模型？
> 它展现了某些世界理解能力，但没有显式的物理引擎。
> 可能是"隐式世界模型"——通过大规模数据学到的。""")

code("""# 世界模型 vs 生成模型 对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 生成模型: 只管生成
ax = axes[0]
ax.set_title('生成模型', fontsize=13)
ax.add_patch(plt.Rectangle((0.1, 0.3), 0.35, 0.4, facecolor='lightblue', edgecolor='black'))
ax.text(0.275, 0.5, '学习\\np(x)\\n数据分布', ha='center', va='center', fontsize=12)
ax.annotate('采样', xy=(0.6, 0.5), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle='->', lw=2))
ax.text(0.75, 0.5, '生成\\n样本', ha='center', va='center', fontsize=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# 世界模型: 理解动力学
ax = axes[1]
ax.set_title('世界模型', fontsize=13)
ax.add_patch(plt.Rectangle((0.05, 0.3), 0.2, 0.4, facecolor='lightyellow', edgecolor='black'))
ax.text(0.15, 0.5, '状态\\n编码', ha='center', va='center', fontsize=11)
ax.add_patch(plt.Rectangle((0.35, 0.3), 0.2, 0.4, facecolor='lightgreen', edgecolor='black'))
ax.text(0.45, 0.5, '动力学\\n预测', ha='center', va='center', fontsize=11)
ax.add_patch(plt.Rectangle((0.65, 0.3), 0.2, 0.4, facecolor='lightcoral', edgecolor='black'))
ax.text(0.75, 0.5, '下一\\n状态', ha='center', va='center', fontsize=11)
ax.annotate('', xy=(0.35, 0.5), xytext=(0.25, 0.5), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(0.65, 0.5), xytext=(0.55, 0.5), arrowprops=dict(arrowstyle='->', lw=2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

plt.tight_layout()
plt.savefig('notebooks/fig_world_model.png', bbox_inches='tight')
plt.show()
print("生成模型学'什么样'; 世界模型学'为什么'。")""")

md("""## 6. 视频生成的演进

```
Make-A-Video (2022)  → DALL·E 扩展到视频
Imagen Video (2022)  → 级联扩散
Stable Video (2023)  → SD + 时间层
VideoLDM (2023)      → 潜空间视频扩散
Sora (2024)          → DiT + 时空 patch
```

### 关键趋势

1. **U-Net → DiT**：Transformer 替代卷积
2. **帧级 → 时空 patch**：统一表示
3. **固定尺寸 → 任意尺寸**：灵活训练
4. **生成 → 世界理解**：涌现能力

> 💡 视频生成正在从"能生成"到"能理解"跨越。
> Sora 是一个里程碑，但远非终点。""")

md("""## 7. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 视频生成三大挑战 | ✅ |
| DiT 架构 | ✅ |
| 时空 patch | ✅ |
| Sora 架构 | ✅ |
| 世界模型概念 | ✅ |

### 核心 takeaway
> **Sora = DiT + 时空 patch + 大数据**。
> 时空 patch 统一了不同视频格式，DiT 提供扩展性。
> Sora 展现了"世界模型"的雏形——隐式理解物理规律。

### 🔗 下一章
**`26_3d_representation.ipynb`** — NeRF/3D Gaussian Splatting

---

> 💬 **板块四进行中。**""")

output_path = "notebooks/25_world_models_video.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")