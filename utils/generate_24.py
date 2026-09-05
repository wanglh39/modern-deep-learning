# 生成 24_vlm.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 24 — VLM：视觉语言模型与原生多模态

> 从 BLIP-2 的 Q-Former 到 LLaVA 的简单投影，再到 GPT-4V 的原生多模态。
> VLM 让 LLM "长出眼睛"。

## 本章你将掌握

1. **BLIP-2**：Q-Former 桥接冻结的视觉编码器和 LLM
2. **LLaVA**：最简单的 VLM——线性投影
3. **原生多模态**：从一开始就联合训练
4. **VLM 评估**：VQA、MMBench""")

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

md("""## 1. VLM 的三种范式

### 1.1 演进路线

```
范式1: 接桥式 (BLIP-2, LLaVA)
  冻结视觉编码器 → 桥接模块 → 冻结 LLM
  优点: 训练便宜; 缺点: 上限受限

范式2: 解冻式 (LLaVA-1.5)
  冻结视觉编码器 → 桥接 → 微调 LLM
  优点: 效果更好; 缺点: 需要更多算力

范式3: 原生多模态 (GPT-4V, Gemini)
  从头联合训练, 视觉和语言共享 backbone
  优点: 效果最好; 缺点: 极其昂贵
```

### 1.2 核心问题

如何把**图像特征**（视觉编码器输出）喂给 **LLM**（期望文本 token）？

```
视觉编码器输出: [N_patches, d_visual]  (如 196 个 patch, 1024 维)
LLM 期望输入:   [L_text, d_llm]        (文本 token, 4096 维)

需要桥接: [N_patches, d_visual] → [M_tokens, d_llm]
```""")

md("""## 2. BLIP-2：Q-Former 桥接

### 2.1 架构

```
图像 → 冻结 ViT → 视觉特征 [196, 1024]
                        ↓
              Q-Former (学习 32 个 query)
                        ↓
              视觉 token [32, 4096]
                        ↓
              冻结 LLM (OPT/FlanT5)
                        ↓
                    文本输出
```

### 2.2 Q-Former

Q-Former 是一个小型 Transformer，有 **32 个可学习的 query token**：

```python
class QFormer:
    queries = nn.Parameter(torch.randn(32, 768))  # 32 个 query

    def forward(visual_features):
        # query 和 visual_features 做交叉注意力
        # query 提取关键视觉信息
        return selected_visual_tokens  # [32, 768]
```

> 💡 Q-Former 像"注意力瓶颈"：32 个 query 从 196 个 patch 中提取最关键的信息，
> 大幅压缩视觉信息（196→32），降低 LLM 的输入负担。""")

code("""# Q-Former 简化实现
class QFormer(nn.Module):
    def __init__(self, n_queries=32, d_query=768, d_visual=1024, n_heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_query) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_query, n_heads, kdim=d_visual, vdim=d_visual, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_query, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_query)
        self.norm2 = nn.LayerNorm(d_query)
        self.ffn = nn.Sequential(nn.Linear(d_query, d_query * 4), nn.GELU(), nn.Linear(d_query * 4, d_query))

    def forward(self, visual_features):
        # visual_features: [batch, n_patches, d_visual]
        batch_size = visual_features.shape[0]
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # 交叉注意力: query 从 visual_features 提取信息
        attn_out, _ = self.cross_attn(queries, visual_features, visual_features)
        queries = self.norm1(queries + attn_out)

        # 自注意力: query 之间交互
        self_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm2(queries + self_out + self.ffn(queries))

        return queries  # [batch, 32, 768]

# 演示 Q-Former
qformer = QFormer(n_queries=32, d_query=768, d_visual=1024)
visual_features = torch.randn(4, 196, 1024)  # 4张图, 196个patch, 1024维
output = qformer(visual_features)
print(f"输入: 视觉特征 {visual_features.shape} (196个patch)")
print(f"输出: Q-Former token {output.shape} (32个query)")
print(f"压缩率: {visual_features.shape[1]} → {output.shape[1]} ({output.shape[1]/visual_features.shape[1]*100:.0f}%)")
print("Q-Former 用 32 个 query 从 196 个 patch 中提取关键信息。")""")

md("""## 3. LLaVA：最简单的 VLM

### 3.1 核心思想

LLaVA 说：**不需要 Q-Former 这么复杂，一个线性投影就够了！**

```
图像 → 冻结 CLIP ViT → 视觉特征 [196, 1024]
                        ↓
              线性投影 W: [1024, 4096]
                        ↓
              视觉 token [196, 4096]
                        ↓
              拼接到文本 token 前面
                        ↓
              LLM (Llama) 处理
```

### 3.2 输入格式

```
USER: <image> 这张图片里有什么？
ASSISTANT: 图片中有一只猫坐在桌子上。
```

`<image>` 位置被替换为 196 个视觉 token。

> 💡 LLaVA 的哲学：**简单就是美**。线性投影 + 指令微调就能达到很好的效果。
> LLaVA-1.5 进一步微调 LLM，效果逼近 GPT-4V。""")

code("""# LLaVA 简化实现
class MiniLLaVA(nn.Module):
    def __init__(self, d_visual=1024, d_llm=4096, n_visual_tokens=196):
        super().__init__()
        # 视觉编码器 (冻结, 这里用随机模拟)
        self.visual_encoder_dim = d_visual
        # 关键: 一个线性投影
        self.projection = nn.Linear(d_visual, d_llm)
        # LLM (简化为一个小 Transformer)
        self.llm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_llm, nhead=8, dim_feedforward=d_llm*4, batch_first=True),
            num_layers=2
        )
        # 输出头
        self.head = nn.Linear(d_llm, 32000)  # 词表大小

    def encode_image(self, image_features):
        # image_features: [batch, n_patches, d_visual]
        return self.projection(image_features)  # [batch, n_patches, d_llm]

    def forward(self, image_features, text_embeddings):
        # 编码图像
        visual_tokens = self.encode_image(image_features)
        # 拼接: [视觉token, 文本token]
        combined = torch.cat([visual_tokens, text_embeddings], dim=1)
        # LLM 处理
        output = self.llm(combined)
        return self.head(output)

# 演示 LLaVA
model = MiniLLaVA(d_visual=512, d_llm=1024, n_visual_tokens=196)
# 模拟视觉特征和文本嵌入
image_features = torch.randn(2, 196, 512)   # 2张图
text_embeddings = torch.randn(2, 10, 1024)  # 10个文本token

visual_tokens = model.encode_image(image_features)
print(f"视觉特征: {image_features.shape} → 视觉token: {visual_tokens.shape}")
print(f"线性投影: {model.projection.in_features}d → {model.projection.out_features}d")

combined = torch.cat([visual_tokens, text_embeddings], dim=1)
print(f"LLM输入: {combined.shape} (视觉token + 文本token)")
print("LLaVA: 线性投影把视觉特征直接喂给 LLM——简单有效!")""")

md("""## 4. 原生多模态：从一开始就联合

### 4.1 思想

不再"接桥"，而是**从头开始**让模型同时处理图像和文本：

```
原生多模态:
  图像 patch → 共享 Transformer ← 文本 token
  (从一开始就混合训练, 视觉和语言共享 backbone)
```

### 4.2 优势

- **统一表示**：视觉和语言在同一空间
- **更深融合**：不是简单拼接，而是层间交互
- **更好泛化**：联合训练学到更丰富的表示

### 4.3 代表模型

| 模型 | 特点 |
|------|------|
| **Gemini** | 从头联合训练 |
| **GPT-4V** | GPT-4 + 视觉 |
| **Qwen-VL** | 阿里原生多模态 |
| **InternVL** | 开源原生多模态 |

> 💡 原生多模态是终极方向，但训练成本极高。
> LLaVA 这类接桥式方法在开源社区更流行——性价比高。""")

code("""# 原生多模态 vs 接桥式 对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 接桥式: 视觉和语言分离
ax = axes[0]
ax.set_title('接桥式 (BLIP-2, LLaVA)', fontsize=13)
boxes = [
    (0.1, 0.7, 0.3, 0.2, '冻结\\n视觉编码器', 'lightblue'),
    (0.1, 0.4, 0.3, 0.2, '桥接模块', 'lightyellow'),
    (0.1, 0.1, 0.3, 0.2, '冻结 LLM', 'lightgreen'),
]
for x, y, w, h, text, color in boxes:
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black'))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11)
ax.annotate('', xy=(0.25, 0.6), xytext=(0.25, 0.7), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(0.25, 0.3), xytext=(0.25, 0.4), arrowprops=dict(arrowstyle='->', lw=2))
ax.text(0.55, 0.4, '• 视觉和语言分离\\n• 只训桥接\\n• 便宜但上限低', fontsize=11, va='center')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# 原生: 联合
ax = axes[1]
ax.set_title('原生多模态 (GPT-4V, Gemini)', fontsize=13)
ax.add_patch(plt.Rectangle((0.1, 0.3), 0.35, 0.4, facecolor='lightcoral', edgecolor='black'))
ax.text(0.275, 0.5, '共享\\nTransformer\\n(联合训练)', ha='center', va='center', fontsize=11)
ax.text(0.55, 0.4, '• 视觉和语言融合\\n• 全参数训练\\n• 贵但效果好', fontsize=11, va='center')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

plt.tight_layout()
plt.savefig('notebooks/fig_vlm_paradigms.png', bbox_inches='tight')
plt.show()
print("接桥式: 便宜但上限低; 原生: 贵但效果好。")""")

md("""## 5. VLM 的训练阶段

### 5.1 典型训练流程

```
阶段1: 视觉-语言对齐预训练
  数据: 图文对 (如 CC3M)
  目标: 让投影矩阵学会把视觉特征对齐到语言空间

阶段2: 视觉指令微调
  数据: (图像, 问题, 回答) 三元组
  目标: 让模型学会看图回答问题
  格式: USER: <image> 问题? ASSISTANT: 回答
```

### 5.2 数据构造

LLaVA 用 GPT-4 自动生成指令数据：

```
输入给 GPT-4: 图片描述 + 物体位置
GPT-4 生成:   "图片中的猫是什么颜色？" → "橘色"
→ 自动得到 (图像, 问题, 回答) 三元组
```

> 💡 用强模型生成训练数据给弱模型——知识蒸馏的变体。""")

code("""# VLM 训练阶段模拟
def vlm_training_simulation():
    # 阶段1: 对齐预训练
    n_align = 10000  # 图文对
    # 阶段2: 指令微调
    n_instruct = 1000  # 指令数据

    # 模拟训练损失下降
    epochs_align = np.arange(50)
    loss_align = 2.0 * np.exp(-epochs_align / 15) + 0.3

    epochs_instruct = np.arange(50, 100)
    loss_instruct = 1.5 * np.exp(-(epochs_instruct - 50) / 20) + 0.1

    all_epochs = np.concatenate([epochs_align, epochs_instruct])
    all_losses = np.concatenate([loss_align, loss_instruct])

    return all_epochs, all_losses

epochs, losses = vlm_training_simulation()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, losses, 'b-', linewidth=2.5)
ax.axvline(50, color='red', linestyle='--', alpha=0.7, label='切换到指令微调')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('VLM 两阶段训练'); ax.legend(); ax.grid(True, alpha=0.3)
ax.annotate('阶段1: 对齐预训练', xy=(25, 1.0), fontsize=12, ha='center', color='navy')
ax.annotate('阶段2: 指令微调', xy=(75, 0.5), fontsize=12, ha='center', color='navy')
plt.tight_layout()
plt.savefig('notebooks/fig_vlm_training.png', bbox_inches='tight')
plt.show()
print("阶段1学对齐, 阶段2学指令——两阶段训练。")""")

md("""## 6. VLM 评估

### 6.1 评估基准

| 基准 | 任务 | 指标 |
|------|------|------|
| **VQAv2** | 看图问答 | 准确率 |
| **GQA** | 场景图问答 | 准确率 |
| **MMBench** | 综合多模态 | 多维度 |
| **MMMU** | 大学级多模态 | 准确率 |
| **MathVista** | 数学推理 | 准确率 |

### 6.2 评估挑战

- **位置理解**：左/右/上/下
- **计数**：图中有几个物体
- **空间关系**：A 在 B 的左边
- **OCR**：图中的文字
- **推理**：结合常识

> 💡 当前 VLM 的弱点：细粒度空间推理、精确计数、复杂图表理解。""")

code("""# VLM 能力雷达图
categories = ['物体识别', 'OCR', '推理', '计数', '空间关系', '图表理解']
n_cats = len(categories)

# 不同模型的能力得分 (0-100)
models_scores = {
    'LLaVA-1.5': [75, 70, 60, 50, 45, 55],
    'GPT-4V':    [95, 90, 85, 75, 80, 80],
    'Gemini':    [93, 88, 88, 78, 82, 85],
}

angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for name, scores in models_scores.items():
    values = scores + scores[:1]
    ax.plot(angles, values, linewidth=2, label=name)
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 100)
ax.set_title('VLM 能力对比', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('notebooks/fig_vlm_radar.png', bbox_inches='tight')
plt.show()
print("GPT-4V/Gemini 各项领先; LLaVA 在细粒度任务上较弱。")""")

md("""## 7. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| VLM 三种范式 | ✅ |
| BLIP-2 Q-Former | ✅ |
| LLaVA 线性投影 | ✅ |
| 原生多模态 | ✅ |
| VLM 训练阶段 | ✅ |
| VLM 评估 | ✅ |

### 核心 takeaway
> **VLM = 视觉编码器 + 桥接 + LLM**。
> 从 BLIP-2 的 Q-Former 到 LLaVA 的线性投影，简单方法往往赢。
> 终极方向是原生多模态，但成本极高。

### 🔗 下一章
**`25_world_models_video.ipynb`** — Sora、视频生成、世界模型

---

> 💬 **板块四进行中。**""")

output_path = "notebooks/24_vlm.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")