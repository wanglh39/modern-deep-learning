# 生成 34_stable_diffusion.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 34 — Stable Diffusion：潜空间扩散与条件生成

> Stable Diffusion 把扩散从像素空间搬到**潜空间**，让生成在消费级 GPU 上可行。
> 文本条件控制让"文字变图片"成为现实。

## 本章你将掌握

1. **Latent Diffusion**：为什么在潜空间扩散
2. **条件机制**：cross-attention 注入文本
3. **Classifier-free Guidance**：增强条件控制
4. **完整流程**：文本→图像""")

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

md("""## 1. Latent Diffusion：潜空间扩散

### 1.1 像素空间的问题

```
像素空间扩散:
  图像 512×512×3 = 786,432 维
  U-Net 在这个空间做扩散 → 计算量巨大

  1000 步 × 786K 维 × U-Net → 需要专业 GPU
```

### 1.2 Latent Diffusion 的解决方案

```
先用 VAE 把图像压缩到潜空间:
  图像 512×512×3 → 潜空间 64×64×4 = 16,384 维
  压缩 48 倍!

在潜空间做扩散:
  U-Net 在 64×64×4 空间工作 → 计算量大幅降低
  → 消费级 GPU 可行
```

### 1.3 架构

```
训练:
  图像 → VAE编码 → 潜变量 z → 在 z 空间扩散训练

生成:
  文本 → 文本编码器 → 条件嵌入
  噪声 z_T → U-Net 去噪 (条件: 文本) → z_0
  z_0 → VAE解码 → 生成图像
```

> 💡 Latent Diffusion 的关键：**VAE 压缩 + 潜空间扩散**。
# 压缩 48 倍让 Stable Diffusion 能在 8GB GPU 上运行—— democratized 生成。""")

code("""# Latent Diffusion 架构模拟
class VAE(nn.Module):
    def __init__(self, in_dim=512*512*3, latent_dim=64*64*4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 4096), nn.ReLU(),
            nn.Linear(4096, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4096), nn.ReLU(),
            nn.Linear(4096, in_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

# 压缩比
img_dim = 512 * 512 * 3
latent_dim = 64 * 64 * 4
print(f"图像空间: {img_dim:,} 维 ({512}×{512}×3)")
print(f"潜空间:   {latent_dim:,} 维 ({64}×{64}×4)")
print(f"压缩比:   {img_dim / latent_dim:.0f}x")
print(f"\\n在潜空间扩散: U-Net 计算量降低 {img_dim / latent_dim:.0f} 倍 → 消费级 GPU 可行。")""")

md("""## 2. 条件机制：Cross-Attention

### 2.1 如何注入文本条件

Stable Diffusion 用 **cross-attention** 把文本注入 U-Net：

```
U-Net 每层:
  自注意力: 图像 patch 之间交互
  交叉注意力: 图像 patch 查询文本 token
    Q = 图像特征
    K, V = 文本嵌入
    → 图像"询问"文本该生成什么
```

### 2.2 文本编码器

```
文本 "一只猫坐在月亮上"
  → CLIP 文本编码器 → [n_tokens, 768] 嵌入
  → 注入 U-Net 的 cross-attention
```

> 💡 Cross-attention 是条件控制的核心——图像特征"查询"文本嵌入。
# 这种机制让文本能精确控制生成内容。""")

code("""# Cross-attention 条件注入
class CrossAttention(nn.Module):
    def __init__(self, d_model=256, d_text=768, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_text, d_model)
        self.v_proj = nn.Linear(d_text, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, image_feat, text_embed):
        # image_feat: [batch, n_img_tokens, d_model]
        # text_embed: [batch, n_text_tokens, d_text]
        B, N_img, D = image_feat.shape
        N_txt = text_embed.shape[1]

        Q = self.q_proj(image_feat).reshape(B, N_img, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(text_embed).reshape(B, N_txt, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(text_embed).reshape(B, N_txt, self.n_heads, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N_img, D)
        return self.out_proj(out)

# 演示
cross_attn = CrossAttention(d_model=256, d_text=768)
image_feat = torch.randn(2, 64, 256)   # 64 个图像 token
text_embed = torch.randn(2, 10, 768)   # 10 个文本 token ("一只猫坐在月亮上")

output = cross_attn(image_feat, text_embed)
print(f"图像特征: {image_feat.shape}")
print(f"文本嵌入: {text_embed.shape}")
print(f"Cross-attention 输出: {output.shape}")
print("文本通过 cross-attention 注入图像生成——条件控制。")""")

md("""## 3. Classifier-free Guidance (CFG)

### 3.1 问题

条件生成有时**忽略条件**——文本说"猫"，但生成的不太像猫。

### 3.2 CFG 解决方案

同时预测**条件**和**无条件**，然后**放大差异**：

$$\\tilde{\\epsilon} = \\epsilon_\\theta(x_t, \\varnothing) + w \\cdot (\\epsilon_\\theta(x_t, c) - \\epsilon_\\theta(x_t, \\varnothing))$$

- $w$：guidance scale（通常 7.5）
- $w=0$：纯无条件
- $w=1$：普通条件
- $w>1$：放大条件影响

### 3.3 效果

```
w=1:  服从条件, 但可能模糊
w=7.5: 更强服从条件, 更清晰, 但可能过度
w=15: 非常强条件, 可能失真
```

> 💡 CFG 是 Stable Diffusion 的标配——没有它，文本控制力很弱。
# guidance scale 7.5 是经验最优值。""")

code("""# Classifier-free Guidance
def cfg_noise_pred(model, x, t, cond, uncond, guidance_scale=7.5):
    # 同时预测条件和 unconditional
    noise_cond = model(x, t, cond)
    noise_uncond = model(x, t, uncond)
    # CFG: 放大条件方向
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)

# 模拟不同 guidance scale 的效果
class SimpleCondModel:
    def __call__(self, x, t, cond):
        if cond is None:
            return torch.randn(3) * 0.5  # 无条件: 随机
        else:
            return torch.randn(3) * 0.3 + cond  # 有条件: 偏向条件

model = SimpleCondModel()
cond = torch.tensor([1.0, 0.5, -0.5])  # "猫" 的条件

scales = [1, 3, 7.5, 15, 30]
fig, ax = plt.subplots(figsize=(10, 5))
for scale in scales:
    preds = []
    for _ in range(100):
        pred = cfg_noise_pred(model, None, None, cond, None, scale)
        preds.append(pred.numpy())
    preds = np.array(preds)
    ax.scatter(preds[:, 0], preds[:, 1], label=f'w={scale}', alpha=0.5, s=10)

ax.scatter([cond[0]], [cond[1]], c='black', marker='*', s=200, label='目标条件')
ax.set_xlabel('维度1'); ax.set_ylabel('维度2')
ax.set_title('Classifier-free Guidance: 不同 scale 的效果')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_cfg.png', bbox_inches='tight')
plt.show()
print("w 越大 → 生成越集中在条件方向 → 更强控制但可能失真。")""")

md("""## 4. Stable Diffusion 完整流程

```
输入: "一只猫坐在月亮上"

1. 文本编码:
   "一只猫坐在月亮上" → CLIP → text_embed [77, 768]

2. 初始化潜变量:
   z_T ~ N(0, I)  [64, 64, 4]

3. 迭代去噪 (50步 DDIM):
   for t in T→0:
     ε_cond = U-Net(z_t, t, text_embed)
     ε_uncond = U-Net(z_t, t, null)
     ε = ε_uncond + 7.5 * (ε_cond - ε_uncond)  # CFG
     z_{t-1} = DDIM_step(z_t, ε)

4. 解码:
   z_0 → VAE解码 → 图像 512×512×3
```

### 4.1 SD 的变体

| 版本 | 特点 |
|------|------|
| **SD 1.4/1.5** | 512×512, CLIP 文本编码器 |
| **SD 2.0/2.1** | 768×768, OpenCLIP |
| **SDXL** | 1024×1024, 双 U-Net |
| **SD3** | DiT 架构, Flow Matching |

> 💡 SD 的演进：更大分辨率 + 更好架构 + 更好训练。
# SD3 转向 DiT + Flow Matching——扩散模型的最新方向。""")

code("""# Stable Diffusion 生成流程模拟
def sd_pipeline_sim(prompt, n_steps=50, guidance_scale=7.5):
    print(f"输入: '{prompt}'")

    # 1. 文本编码 (模拟)
    text_embed = torch.randn(77, 768)
    print(f"1. 文本编码: '{prompt}' → {text_embed.shape}")

    # 2. 初始化潜变量
    z = torch.randn(1, 4, 64, 64)
    print(f"2. 初始化潜变量: {z.shape}")

    # 3. 去噪 (模拟)
    losses = []
    for t in range(n_steps):
        # 模拟去噪: loss 下降
        loss = 2.0 * np.exp(-t / 15) + 0.1
        losses.append(loss)

    # 4. VAE 解码 (模拟)
    image = torch.randn(3, 512, 512)
    print(f"3. 去噪: {n_steps} 步, guidance={guidance_scale}")
    print(f"4. VAE 解码: {z.shape} → {image.shape}")

    return losses

losses = sd_pipeline_sim("一只猫坐在月亮上", n_steps=50)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(losses, 'b-', linewidth=2)
ax.set_xlabel('去噪步数'); ax.set_ylabel('噪声水平')
ax.set_title('Stable Diffusion 去噪过程 (50步 DDIM)'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_sd_pipeline.png', bbox_inches='tight')
plt.show()
print("\\n完整流程: 文本→编码→潜空间去噪→VAE解码→图像。")""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| Latent Diffusion | ✅ |
| VAE 压缩 | ✅ |
| Cross-attention 条件 | ✅ |
| Classifier-free Guidance | ✅ |
| SD 完整流程 | ✅ |

### 核心 takeaway
> **Stable Diffusion = VAE + 潜空间扩散 + 文本条件 + CFG**。
> 潜空间压缩让生成在消费级 GPU 可行，
> cross-attention + CFG 让文本精确控制生成。

### 🔗 下一章
**`35_controlnet.ipynb`** — ControlNet/IP-Adapter 条件控制

---

> 💬 **板块六进行中。**""")

output_path = "notebooks/34_stable_diffusion.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")