# 生成 35_controlnet.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 35 — ControlNet / IP-Adapter：精确条件控制

> 文本控制不够精确——"一只猫"不能控制猫的姿势。
> ControlNet 用边缘/深度/姿态精确控制生成，IP-Adapter 用参考图控制风格。

## 本章你将掌握

1. **ControlNet**：零卷积 + 副本分支
2. **控制类型**：Canny/Depth/Pose/OpenPose
3. **IP-Adapter**：参考图注入
4. **组合控制**：多条件叠加""")

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

md("""## 1. 为什么需要 ControlNet

### 1.1 文本控制的局限

```
文本: "一个人站着" → 能生成人, 但不能控制:
  - 具体姿势 (手举多高? 站还是坐?)
  - 空间位置 (在画面哪里?)
  - 场景结构 (背景是什么?)
```

### 1.2 ControlNet 的思想

用**结构化条件**（边缘图、深度图、姿态图）精确控制：

```
文本: "一个人" + Canny边缘图 → 生成符合边缘的人
文本: "一个房间" + 深度图 → 生成符合深度的房间
文本: "一个人跳舞" + OpenPose → 生成符合姿态的人
```

> 💡 ControlNet 让 Stable Diffusion 从"文字描述生成"变成"结构控制生成"。
# 这是 AI 绘画从玩具到工具的关键一步。""")

md("""## 2. ControlNet 架构

### 2.1 核心设计

```
ControlNet = U-Net 副本 + 零卷积

原始 U-Net (冻结):
  输入: z_t, text_embed
  输出: ε

ControlNet 分支 (可训练):
  输入: z_t, text_embed, control_image (边缘/深度/姿态)
  输出: Δε (残差)

最终: ε + Δε → 加了结构控制
```

### 2.2 零卷积

ControlNet 的关键：**零卷积**初始化。

```
零卷积: 权重和偏置都初始化为 0
  → 训练开始时 Δε = 0
  → ControlNet 不改变原始 SD 的行为
  → 逐渐学习如何注入控制信号
```

### 2.3 为什么这样设计

```
1. 复制 U-Net: 保留 SD 的全部能力
2. 冻结原始: 不破坏预训练知识
3. 零卷积: 平滑引入控制, 训练稳定
4. 残差注入: 控制是"修正"而非"替代"
```

> 💡 零卷积是 ControlNet 的天才设计——
# 初始时 ControlNet 完全不影响 SD，然后逐渐学会注入控制。""")

code("""# ControlNet 简化实现
class ZeroConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # 零初始化
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

class ControlNetBlock(nn.Module):
    def __init__(self, d_model=256, control_channels=3):
        super().__init__()
        # 控制图像编码
        self.control_encoder = nn.Sequential(
            nn.Conv2d(control_channels, d_model, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
        )
        # 副本分支 (模拟 U-Net 的一层)
        self.copy_block = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
        )
        # 零卷积输出
        self.zero_conv = ZeroConv(d_model, d_model)

    def forward(self, control_image):
        # 编码控制图像
        control_feat = self.control_encoder(control_image)
        # 通过副本分支
        hidden = self.copy_block(control_feat)
        # 零卷积 → 残差
        delta = self.zero_conv(hidden)
        return delta

# 演示 ControlNet
controlnet = ControlNetBlock(d_model=64, control_channels=3)

# 不同控制类型
control_types = {
    'Canny边缘': torch.randn(1, 3, 32, 32),
    '深度图': torch.randn(1, 3, 32, 32),
    'OpenPose': torch.randn(1, 3, 32, 32),
}

print("ControlNet 不同控制类型:")
for name, control in control_types.items():
    delta = controlnet(control)
    print(f"  {name}: {control.shape} → Δε {delta.shape}, 初始 Δε ≈ {delta.abs().mean().item():.6f}")

print("\\n零卷积: 初始 Δε ≈ 0, 不破坏原始 SD。")""")

md("""## 3. 控制类型

### 3.1 常见控制

| 控制类型 | 提取方法 | 控制什么 |
|---------|---------|---------|
| **Canny** | 边缘检测 | 轮廓/形状 |
| **Depth** | 深度估计 | 3D 结构 |
| **OpenPose** | 姿态估计 | 人体姿势 |
| **HED** | 软边缘 | 边界 |
| **Scribble** | 涂鸦 | 粗略形状 |
| **Segmentation** | 分割 | 区域 |
| **Normal** | 法线 | 表面方向 |

### 3.2 多控制组合

ControlNet 可以**同时用多个控制**：

```
文本 + Canny + Depth + OpenPose → 同时控制形状+深度+姿态

组合方式: Δε = Σ_i w_i * ControlNet_i(control_i)
```

> 💡 多控制组合让生成精确到每个细节——
# 既能控制整体结构，又能控制人体姿态。""")

code("""# 多控制组合
class MultiControlNet:
    def __init__(self, control_types=['canny', 'depth', 'pose']):
        self.controls = {name: ControlNetBlock(d_model=64) for name in control_types}

    def forward(self, control_images, weights=None):
        if weights is None:
            weights = {name: 1.0 for name in control_images}

        total_delta = 0
        for name, img in control_images.items():
            delta = self.controls[name](img)
            total_delta += weights[name] * delta
        return total_delta

multi_ctrl = MultiControlNet()
controls = {
    'canny': torch.randn(1, 3, 32, 32),
    'depth': torch.randn(1, 3, 32, 32),
    'pose': torch.randn(1, 3, 32, 32),
}

# 不同权重组合
configs = [
    {'canny': 1.0, 'depth': 0.5, 'pose': 0.8},  # 均衡
    {'canny': 2.0, 'depth': 0.0, 'pose': 0.0},  # 只用边缘
    {'canny': 0.0, 'depth': 0.0, 'pose': 2.0},  # 只用姿态
]

for i, weights in enumerate(configs):
    delta = multi_ctrl.forward(controls, weights)
    print(f"配置{i+1} {weights}: Δε 范数 = {delta.norm().item():.4f}")

print("\\n多控制组合: 不同权重 → 不同控制强度。")""")

md("""## 4. IP-Adapter：参考图注入

### 4.1 动机

ControlNet 控制结构，但**风格/内容**控制不够。
IP-Adapter 用参考图直接控制生成风格。

### 4.2 架构

```
参考图 → CLIP 图像编码器 → image_embed [n_tokens, d]
                              ↓
注入 U-Net 的 cross-attention:
  原始: Q=图像, K/V=文本
  IP-Adapter: Q=图像, K/V=文本 + K/V=image_embed
  → 图像可以"查询"参考图
```

### 4.3 应用

```
参考图: 某画家的作品 → 生成该风格的新图
参考图: 某产品照片 → 生成该产品在新场景
参考图: 某人物 → 生成该人物在不同姿态
```

> 💡 IP-Adapter 比 ControlNet 轻量——只加一个 cross-attention 模块。
# 适合风格/内容迁移，与 ControlNet 互补。""")

code("""# IP-Adapter 简化
class IPAdapter(nn.Module):
    def __init__(self, d_model=256, d_image=768):
        super().__init__()
        # 图像嵌入投影
        self.image_proj = nn.Linear(d_image, d_model)
        # 额外的 cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, kdim=d_model, vdim=d_model, batch_first=True)

    def forward(self, image_feat, image_embed):
        # 投影参考图嵌入
        image_tokens = self.image_proj(image_embed)  # [B, N, d]
        # 图像特征查询参考图
        attn_out, _ = self.cross_attn(image_feat, image_tokens, image_tokens)
        return attn_out

# 演示
ip_adapter = IPAdapter(d_model=256, d_image=768)
image_feat = torch.randn(2, 64, 256)   # 生成中的图像特征
ref_embed = torch.randn(2, 10, 768)    # 参考图的 CLIP 嵌入

output = ip_adapter(image_feat, ref_embed)
print(f"参考图嵌入: {ref_embed.shape}")
print(f"图像特征: {image_feat.shape}")
print(f"IP-Adapter 输出: {output.shape}")
print("IP-Adapter: 参考图通过 cross-attention 注入——风格/内容控制。")""")

md("""## 5. ControlNet vs IP-Adapter

| 特性 | ControlNet | IP-Adapter |
|------|-----------|-----------|
| **控制类型** | 结构 (边缘/深度/姿态) | 风格/内容 |
| **计算开销** | 大 (复制 U-Net) | 小 (加 attention) |
| **训练数据** | 配对 (控制图, 图像) | 配对 (参考图, 图像) |
| **组合使用** | ✅ 多个叠加 | ✅ 与 ControlNet 叠加 |

### 最佳实践

```
精确控制: ControlNet (Canny + Depth + Pose)
风格迁移: IP-Adapter (参考图)
两者结合: ControlNet 控结构 + IP-Adapter 控风格
→ 既能控制姿势, 又能控制画风
```

> 💡 ControlNet + IP-Adapter 是当前 AI 绘画的黄金组合——
# 结构和风格都能精确控制。""")

code("""# ControlNet + IP-Adapter 组合
def combined_control(controlnet, ip_adapter, control_image, ref_embed, image_feat, text_embed):
    # ControlNet: 结构控制
    delta_structure = controlnet(control_image)

    # IP-Adapter: 风格控制
    delta_style = ip_adapter(image_feat, ref_embed)

    # 组合: 结构 + 风格
    # 实际中通过不同方式注入 (ControlNet 加到 U-Net 输出, IP-Adapter 加到 attention)
    return delta_structure, delta_style

# 演示
control_img = torch.randn(1, 3, 32, 32)
ref_embed = torch.randn(1, 10, 768)
image_feat = torch.randn(1, 64, 256)
text_embed = torch.randn(1, 10, 768)

delta_struct, delta_style = combined_control(
    controlnet, ip_adapter, control_img, ref_embed, image_feat, text_embed
)

print(f"ControlNet (结构): {delta_struct.shape}")
print(f"IP-Adapter (风格): {delta_style.shape}")
print("组合: ControlNet 控结构 + IP-Adapter 控风格 → 精确控制生成。")

# 可视化控制类型
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
titles = ['Canny 边缘', '深度图', 'OpenPose 姿态', '参考图 (IP-Adapter)']
for ax, title in zip(axes, titles):
    ax.imshow(np.random.rand(64, 64), cmap='gray')
    ax.set_title(title, fontsize=11); ax.axis('off')
plt.suptitle('ControlNet + IP-Adapter: 结构 + 风格控制', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/fig_controlnet.png', bbox_inches='tight')
plt.show()""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| ControlNet 架构 | ✅ |
| 零卷积初始化 | ✅ |
| 控制类型 (Canny/Depth/Pose) | ✅ |
| 多控制组合 | ✅ |
| IP-Adapter 参考图 | ✅ |

### 核心 takeaway
> **ControlNet 用零卷积注入结构控制，IP-Adapter 用参考图注入风格控制**。
# 两者组合让 AI 绘画精确到结构和风格——从玩具到工具。

### 🔗 下一章
**`36_flow_matching.ipynb`** — Flow Matching/Rectified Flow

---

> 💬 **板块六进行中。**""")

output_path = "notebooks/35_controlnet.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")