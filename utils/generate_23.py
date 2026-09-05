# 生成 23_clip_blip.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 23 — CLIP / BLIP：多模态对齐

> 让模型**看懂图片**——CLIP 开创对比学习对齐，BLIP 进一步支持生成。
> 这是所有 VLM（GPT-4V、Gemini）的起点。

## 本章你将掌握

1. **CLIP 原理**：双塔对比学习，图文共享嵌入空间
2. **InfoNCE 损失**：对比学习的核心
3. **零样本分类**：不需要训练就能分类
4. **BLIP**：理解 + 生成的统一框架""")

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

md("""## 1. CLIP：对比语言-图像预训练

### 1.1 核心思想

CLIP（Contrastive Language-Image Pre-training）用 **4 亿图文对** 训练，
让图像和文本在同一个嵌入空间中对齐：

```
图像编码器 ImageEncoder     文本编码器 TextEncoder
    ↓                           ↓
图像嵌入 [I1, I2, ..., IN]    文本嵌入 [T1, T2, ..., TN]

目标: 匹配的 (Ii, Ti) 靠近, 不匹配的 (Ii, Tj) 远离
```

### 1.2 双塔架构

```
图像塔:  图片 → ViT/ResNet → 图像嵌入 (d维)
文本塔:  文本 → Transformer → 文本嵌入 (d维)

两个嵌入在同一个 d 维空间中, 用余弦相似度比较
```

> 💡 关键：**对比学习**不需要类别标签，只需要"这张图配这段文字"的配对信号。
> 4 亿图文对从互联网爬取，规模远超 ImageNet 的 130 万。""")

md("""## 2. InfoNCE 损失：对比学习的核心

### 2.1 公式

给定一个 batch 的 N 对 (图像, 文本)，InfoNCE 损失：

$$\\mathcal{L} = -\\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{\\exp(s(I_i, T_i) / \\tau)}{\\sum_{j=1}^{N} \\exp(s(I_i, T_j) / \\tau)}$$

- $s(I, T) = \\frac{I \\cdot T}{|I| |T|}$ 是余弦相似度
- $\\tau$ 是温度参数（可学习）
- 分子：正样本对（匹配的图文）
- 分母：所有可能配对（1 个正 + N-1 个负）

### 2.2 对称损失

CLIP 同时做两个方向的对比：
- 图 → 文：给定图像，找匹配的文本
- 文 → 图：给定文本，找匹配的图像

$$\\mathcal{L}_{CLIP} = \\frac{1}{2}(\\mathcal{L}_{I \\to T} + \\mathcal{L}_{T \\to I})$$

> 💡 温度 $\\tau$ 越小，对比越"硬"（只关注最相似的）；越大越"软"。
> CLIP 中 $\\tau$ 是可学习参数，初始化为 0.07。""")

code("""# InfoNCE 损失实现
def clip_loss(image_embeds, text_embeds, temperature=0.07):
    # 归一化嵌入
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 相似度矩阵 [N, N]
    logits = (image_embeds @ text_embeds.T) / temperature

    # 对称损失
    labels = torch.arange(len(image_embeds))
    loss_i2t = F.cross_entropy(logits, labels)  # 图→文
    loss_t2i = F.cross_entropy(logits.T, labels)  # 文→图
    return (loss_i2t + loss_t2i) / 2

# 演示: 4对图文, 嵌入维度 64
N, d = 4, 64
# 模拟: 匹配的图文嵌入接近, 不匹配的远离
torch.manual_seed(42)
base = torch.randn(N, d)
image_embeds = base + 0.1 * torch.randn(N, d)  # 图像嵌入
text_embeds = base + 0.1 * torch.randn(N, d)   # 文本嵌入 (与图像接近)

loss_matched = clip_loss(image_embeds, text_embeds, temperature=0.07)
print(f"匹配的图文对: CLIP loss = {loss_matched:.4f}")

# 打乱配对
perm = torch.randperm(N)
loss_mismatched = clip_loss(image_embeds, text_embeds[perm], temperature=0.07)
print(f"打乱配对后:   CLIP loss = {loss_mismatched:.4f}")
print(f"\\n匹配的 loss 更低 → 对比学习让匹配的图文靠近。")""")

md("""## 3. 从零实现迷你 CLIP

### 3.1 架构

```
ImageEncoder:  图片特征 (512维) → MLP → 嵌入 (128维)
TextEncoder:   文本特征 (512维) → MLP → 嵌入 (128维)
```""")

code("""# 迷你 CLIP 实现
class MiniCLIP(nn.Module):
    def __init__(self, feat_dim=512, embed_dim=128):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Linear(256, embed_dim)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.GELU(), nn.Linear(256, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image_feats):
        return F.normalize(self.image_encoder(image_feats), dim=-1)

    def encode_text(self, text_feats):
        return F.normalize(self.text_encoder(text_feats), dim=-1)

    def forward(self, image_feats, text_feats):
        image_embeds = self.encode_image(image_feats)
        text_embeds = self.encode_text(text_feats)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text

# 训练迷你 CLIP
def train_mini_clip(n_epochs=100, n_pairs=64, feat_dim=512, embed_dim=128):
    model = MiniCLIP(feat_dim, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 模拟图文特征: 匹配的对有共享信息
    torch.manual_seed(42)
    shared = torch.randn(n_pairs, feat_dim)
    image_feats = shared + 0.3 * torch.randn(n_pairs, feat_dim)
    text_feats = shared + 0.3 * torch.randn(n_pairs, feat_dim)

    losses = []
    for epoch in range(n_epochs):
        logits_i, logits_t = model(image_feats, text_feats)
        labels = torch.arange(n_pairs)
        loss = (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels)) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses

model, losses = train_mini_clip()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('CLIP Loss')
ax.set_title('迷你 CLIP 训练'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_clip_training.png', bbox_inches='tight')
plt.show()
print(f"训练完成: loss {losses[0]:.3f} → {losses[-1]:.3f}")""")

md("""## 4. 零样本分类

CLIP 最神奇的能力：**不需要训练就能分类**。

### 4.1 方法

```
类别: "狗", "猫", "鸟"
提示模板: "a photo of a {class}"

文本嵌入: encode("a photo of a dog"), encode("a photo of a cat"), ...
图像嵌入: encode(图片)

预测: argmax_i cosine_sim(图像嵌入, 文本嵌入_i)
```

不需要任何标注数据，只需要类别名称！""")

code("""# 零样本分类演示
def zero_shot_classify(model, image_feat, class_names, template="a photo of a {}"):
    # 模拟文本特征 (实际中用文本编码器)
    n_classes = len(class_names)
    torch.manual_seed(42)
    # 每个类别有一个"原型"特征
    class_prototypes = torch.randn(n_classes, model.image_encoder[0].in_features)

    with torch.no_grad():
        image_embed = model.encode_image(image_feat.unsqueeze(0))
        text_embeds = model.encode_text(class_prototypes)
        similarities = (image_embed @ text_embeds.T).squeeze()
    return similarities

# 模拟: 5个类别, 测试图片
class_names = ['狗', '猫', '鸟', '鱼', '马']
torch.manual_seed(123)
# 图片特征最接近"狗"的原型
image_feat = torch.randn(model.image_encoder[0].in_features)
image_feat[:100] += 2.0  # 偏向某个方向

sims = zero_shot_classify(model, image_feat, class_names)
probs = F.softmax(sims * 10, dim=0)  # 放大后 softmax

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(class_names, probs.numpy(), color='steelblue', edgecolor='navy')
ax.set_ylabel('概率'); ax.set_title('CLIP 零样本分类')
ax.set_ylim(0, max(probs.numpy()) * 1.3)
for bar, prob in zip(bars, probs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{prob:.2f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('notebooks/fig_zero_shot.png', bbox_inches='tight')
plt.show()
print(f"预测类别: {class_names[probs.argmax()]}")
print("零样本分类: 不需要训练数据, 只需要类别名称!")""")

md("""## 5. BLIP：理解 + 生成

### 5.1 CLIP 的局限

CLIP 只能做**理解**（检索、分类），不能**生成**（看图说话）。

### 5.2 BLIP 的三个头

```
共享图像编码器
    ↓
┌───────────┬───────────┐
│ ITC 头    │ ITM 头    │ LM 头
│ (对比)   │ (匹配)   │ (生成)
│ 图文检索 │ 图文匹配 │ 看图说话
└───────────┴───────────┘
```

| 头 | 任务 | 损失 |
|----|------|------|
| ITC | 图文对比 (同 CLIP) | InfoNCE |
| ITM | 图文匹配 (二分类) | 交叉熵 |
| LM | 条件语言模型 | 交叉熵 (自回归) |

### 5.3 自举数据清洗

BLIP 的另一创新：**自动清洗噪声数据**。

```
1. 用初始模型给图文对打分
2. 高分 → 保留 (好的配对)
3. 低分 → 丢弃 (噪声配对)
4. 用清洗后的数据重新训练
→ 迭代提升数据质量
```

> 💡 互联网图文对很多是噪声（alt text 与图片不匹配）。
> BLIP 用模型自己判断配对质量，自举清洗——数据飞轮。""")

code("""# BLIP 自举数据清洗模拟
def blip_bootstrap(n_pairs=200, n_iterations=5):
    # 模拟: 70% 是好的配对, 30% 是噪声
    torch.manual_seed(42)
    true_quality = np.random.binomial(1, 0.7, n_pairs)  # 1=好, 0=噪声

    # 模型预测质量 (初始准确率 70%, 逐渐提升)
    qualities = []
    data_quality_history = []

    for it in range(n_iterations):
        # 模型预测: 准确率随迭代提升
        acc = 0.7 + 0.05 * it
        pred_quality = (true_quality * np.random.binomial(1, acc, n_pairs) +
                       (1 - true_quality) * np.random.binomial(1, 1 - acc, n_pairs))

        # 保留预测为"好"的
        kept = pred_quality == 1
        precision = true_quality[kept].mean() if kept.sum() > 0 else 0
        recall = true_quality[kept].sum() / true_quality.sum()

        qualities.append((precision, recall, kept.sum()))
        data_quality_history.append(true_quality[kept].mean() if kept.sum() > 0 else 0)

    return qualities, data_quality_history

qualities, dq_history = blip_bootstrap()
print("BLIP 自举数据清洗:")
print(f"{'迭代':>4s} {'精确率':>8s} {'召回率':>8s} {'保留数':>8s} {'数据质量':>8s}")
for i, (p, r, k) in enumerate(qualities):
    print(f"{i:4d} {p:8.3f} {r:8.3f} {k:8d} {dq_history[i]:8.3f}")
print("\\n迭代后数据质量提升 → 模型质量提升 → 数据质量提升 (飞轮)")""")

md("""## 6. 图文检索演示

CLIP 最直接的应用：给一段文字，找最匹配的图片（或反过来）。""")

code("""# 图文检索演示
def image_text_retrieval(model, n_images=8, n_texts=8):
    torch.manual_seed(42)
    # 模拟图像和文本特征
    shared = torch.randn(n_images, model.image_encoder[0].in_features)
    image_feats = shared + 0.2 * torch.randn(n_images, model.image_encoder[0].in_features)
    text_feats = shared[:n_texts] + 0.2 * torch.randn(n_texts, model.image_encoder[0].in_features)

    with torch.no_grad():
        image_embeds = model.encode_image(image_feats)
        text_embeds = model.encode_text(text_feats)
        sim_matrix = image_embeds @ text_embeds.T

    return sim_matrix.numpy()

sim_matrix = image_text_retrieval(model)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ax.set_xlabel('文本索引'); ax.set_ylabel('图像索引')
ax.set_title('图文相似度矩阵 (对角线=匹配)')
plt.colorbar(im, ax=ax, label='余弦相似度')

# 标记对角线
for i in range(min(sim_matrix.shape)):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=2))

plt.tight_layout()
plt.savefig('notebooks/fig_retrieval.png', bbox_inches='tight')
plt.show()
print("对角线(匹配对)相似度最高 → CLIP 学会了图文对齐。")""")

md("""## 7. CLIP vs BLIP 对比

| 特性 | CLIP | BLIP |
|------|------|------|
| **任务** | 理解 (检索/分类) | 理解 + 生成 |
| **损失** | ITC | ITC + ITM + LM |
| **架构** | 双塔 | 共享图像编码器 + 多头 |
| **数据** | 4亿对 | 自举清洗 |
| **生成** | ❌ | ✅ (看图说话) |
| **下游** | 零样本分类 | + VQA, 图像描述 |

### 演进路线

```
CLIP (2021)     → 对比学习, 零样本
BLIP (2022)     → + 生成, 自举清洗
BLIP-2 (2023)   → + Q-Former, 接 LLM
LLaVA (2023)    → 简化 BLIP-2, 开源
GPT-4V (2023)   → 原生多模态
```

> 💡 从 CLIP 到 GPT-4V 的主线：**对齐 → 生成 → 原生多模态**。""")

md("""## 8. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| CLIP 双塔对比学习 | ✅ |
| InfoNCE 对比损失 | ✅ |
| 零样本分类 | ✅ |
| BLIP 理解+生成 | ✅ |
| 自举数据清洗 | ✅ |

### 核心 takeaway
> **CLIP 用对比学习把图文对齐到同一空间**，开启了多模态时代。
> BLIP 加上生成能力和数据自举，成为 VLM 的基石。
> 从 CLIP 到 GPT-4V，主线是对齐 → 生成 → 原生多模态。

### 🔗 下一章
**`24_vlm.ipynb`** — VLM、原生多模态（BLIP-2/LLaVA/GPT-4V）

---

> 💬 **进入板块四(多模态与具身)。**""")

output_path = "notebooks/23_clip_blip.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")