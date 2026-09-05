# 生成 30_contrastive_learning.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 30 — 对比学习：SimCLR、MoCo、InfoNCE

> 自监督学习让模型不需要标签就能学好表示。
> 对比学习是其中最重要的范式——"拉近正样本，推开负样本"。

## 本章你将掌握

1. **自监督学习思想**：为什么要自监督
2. **SimCLR**：简单对比学习框架
3. **MoCo**：动量编码器 + 队列
4. **InfoNCE 损失**：对比学习的数学""")

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

md("""## 1. 自监督学习：为什么要自监督

### 1.1 标签瓶颈

```
监督学习: 需要人工标注
  ImageNet: 130万张图, 1000类 → 贵
  GPT-4 级别: 需要百万级标注 → 极贵

自监督学习: 从数据自身构造标签
  图像: 同一图的不同增强 = 正样本
  文本: 遮盖 token, 预测它 (MLM)
  → 不需要人工标注, 用海量无标签数据
```

### 1.2 对比学习的直觉

```
给一张猫的图片:
  增强1: 翻转 + 裁剪 → 还是猫
  增强2: 颜色变换 → 还是猫
  其他图片: 狗、车、树 → 不是猫

目标: 让猫的两个增强靠近, 猫和狗/车/树远离
→ 学到的表示能区分不同物体
```

> 💡 对比学习的核心：**构造正负样本对，拉近正、推开负**。
# 不需要标签，只需要知道"这两个是同类的"。""")

md("""## 2. SimCLR：简单对比学习

### 2.1 流程

```
图片 x
  → 增强 t1 → x1 → 编码器 f → h1 → 投影头 g → z1
  → 增强 t2 → x2 → 编码器 f → h2 → 投影头 g → z2

z1, z2 = 正样本对
其他图片的 z = 负样本

损失: InfoNCE(z1, z2, {负样本})
```

### 2.2 数据增强

SimCLR 用以下增强构造正样本：
- 随机裁剪 + 缩放
- 颜色抖动（亮度/对比度/饱和度/色相）
- 随机翻转

### 2.3 投影头

```
编码器 f: 输出表示 h (用于下游任务)
投影头 g: h → z (用于对比损失)

关键: 投影头只在训练时用, 下游任务用 h
```

> 💡 SimCLR 的发现：**投影头很重要**——它让表示空间和对比空间解耦。
# 下游任务用 h（不经过投影头）效果更好。""")

code("""# SimCLR 简化实现
class SimCLR(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=256, proj_dim=64):
        super().__init__()
        # 编码器 (简化)
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, x):
        h = self.encoder(x)  # 表示
        z = self.projector(h)  # 投影
        return F.normalize(z, dim=-1)

def simclr_loss(z1, z2, temperature=0.5):
    # InfoNCE 损失
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2N, d]
    sim = z @ z.T / temperature  # [2N, 2N]

    # 掩码对角线
    mask = torch.eye(2 * batch_size, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)

    # 正样本: (i, i+N) 和 (i+N, i)
    labels = torch.cat([torch.arange(batch_size, 2*batch_size),
                        torch.arange(0, batch_size)])
    loss = F.cross_entropy(sim, labels)
    return loss

# 训练 SimCLR
def train_simclr(n_epochs=50, batch_size=64):
    model = SimCLR()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 模拟数据: 同类样本接近
    torch.manual_seed(42)
    centers = torch.randn(10, 784)  # 10个类
    losses = []
    for _ in range(n_epochs):
        # 生成正样本对
        idx = torch.randint(0, 10, (batch_size,))
        x1 = centers[idx] + 0.3 * torch.randn(batch_size, 784)
        x2 = centers[idx] + 0.3 * torch.randn(batch_size, 784)

        z1 = model(x1)
        z2 = model(x2)
        loss = simclr_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses

model, losses = train_simclr()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('InfoNCE Loss')
ax.set_title('SimCLR 训练'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_simclr_train.png', bbox_inches='tight')
plt.show()
print(f"SimCLR 训练: loss {losses[0]:.3f} → {losses[-1]:.3f}")""")

md("""## 3. MoCo：动量编码器 + 队列

### 3.1 SimCLR 的问题

SimCLR 需要 **大 batch**（4096）才能有足够负样本：
- batch 内所有其他样本作为负样本
- 大 batch 需要大量显存

### 3.2 MoCo 的解决方案

MoCo 用 **队列** 存储负样本，不需要大 batch：

```
查询单元 f_q:  编码当前 batch (有梯度, 正常更新)
键单元 f_k:    动量更新 (不回传梯度)

队列 queue:    存储之前的 f_k 输出
  → 负样本从队列取, 不依赖 batch 大小

动量更新: f_k.params = m * f_k.params + (1-m) * f_q.params
  (m=0.999, 慢更新, 保持一致性)
```

### 3.3 MoCo v1/v2/v3

```
MoCo v1: 队列 + 动量编码器
MoCo v2: + SimCLR 的投影头 + 更强增强
MoCo v3: + ViT 架构 (从 CNN 到 Transformer)
```

> 💡 MoCo 的智慧：**队列解耦负样本数量和 batch 大小**。
# 小 batch (256) 也能有大量负样本 (65536)——省显存。""")

code("""# MoCo 简化实现
class MoCo(nn.Module):
    def __init__(self, feat_dim=128, queue_size=65536, momentum=0.999):
        super().__init__()
        self.momentum = momentum
        self.queue_size = queue_size

        # 查询编码器
        self.encoder_q = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, feat_dim)
        )
        # 键编码器 (动量更新)
        self.encoder_k = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, feat_dim)
        )
        # 初始化键编码器 = 查询编码器
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 队列
        self.register_buffer('queue', torch.randn(queue_size, feat_dim))
        self.queue = F.normalize(self.queue, dim=-1)

    @torch.no_grad()
    def momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    def forward(self, x_q, x_k):
        q = F.normalize(self.encoder_q(x_q), dim=-1)
        with torch.no_grad():
            k = F.normalize(self.encoder_k(x_k), dim=-1)
        return q, k

    def loss(self, q, k, temperature=0.07):
        # 正样本: q · k
        l_pos = (q * k).sum(dim=-1, keepdim=True) / temperature  # [N, 1]
        # 负样本: q · queue
        l_neg = q @ self.queue.T / temperature  # [N, K]
        # InfoNCE
        logits = torch.cat([l_pos, l_neg], dim=1)  # [N, 1+K]
        labels = torch.zeros(q.shape[0], dtype=torch.long)  # 正样本在位置0
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def enqueue_dequeue(self, k):
        # 入队新键, 出队旧键
        batch_size = k.shape[0]
        self.queue = torch.cat([k, self.queue[:-batch_size]], dim=0)

# 演示 MoCo
moco = MoCo(feat_dim=64, queue_size=1024, momentum=0.99)
x_q = torch.randn(32, 784)
x_k = torch.randn(32, 784)
q, k = moco(x_q, x_k)
loss = moco.loss(q, k)
moco.enqueue_dequeue(k)
moco.momentum_update()

print(f"队列大小: {moco.queue_size}")
print(f"batch 大小: 32")
print(f"负样本数: {moco.queue_size} (来自队列, 不依赖 batch)")
print(f"Loss: {loss.item():.3f}")
print("MoCo: 队列解耦负样本和 batch 大小——小 batch 也能训练。")""")

md("""## 4. InfoNCE 损失详解

### 4.1 公式

$$\\mathcal{L} = -\\log \\frac{\\exp(s(z_i, z_j) / \\tau)}{\\sum_{k=0}^{K} \\exp(s(z_i, z_k) / \\tau)}$$

- $z_i$: 查询样本
- $z_j$: 正样本
- $z_k$: 所有候选（1 正 + K 负）
- $s$: 余弦相似度
- $\\tau$: 温度

### 4.2 温度的作用

```
τ 小 → 分布尖锐 → 只关注最相似的 → 学得"硬"
τ 大 → 分布平坦 → 所有样本都考虑 → 学得"软"
```

### 4.3 与互信息的关系

InfoNCE 是互信息的**下界估计**：

$$\\mathcal{L}_{NCE} \\geq \\log(K) - I(z_i, z_j)$$

- $K$: 负样本数
- $I$: 互信息
- 更多负样本 → 更紧的下界 → 更好的表示

> 💡 InfoNCE 的理论：最大化正样本的互信息下界。
# 负样本越多，下界越紧，表示越好——这就是为什么 MoCo 要大队列。""")

code("""# 温度参数影响
def info_nce_with_temp(z1, z2, temperature):
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temperature
    mask = torch.eye(z.shape[0], dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)
    labels = torch.cat([torch.arange(z1.shape[0], 2*z1.shape[0]),
                        torch.arange(0, z1.shape[0])])
    return F.cross_entropy(sim, labels).item()

# 生成数据
torch.manual_seed(42)
z1 = F.normalize(torch.randn(64, 32), dim=-1)
z2 = F.normalize(z1 + 0.1 * torch.randn(64, 32), dim=-1)

temperatures = np.logspace(-2, 1, 50)
losses = [info_nce_with_temp(z1, z2, t) for t in temperatures]

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(temperatures, losses, 'b-', linewidth=2.5)
ax.set_xlabel('温度 τ'); ax.set_ylabel('InfoNCE Loss')
ax.set_title('温度参数对 InfoNCE 的影响'); ax.grid(True, alpha=0.3)
ax.axvline(0.07, color='red', linestyle='--', alpha=0.7, label='SimCLR 默认 (0.07)')
ax.legend()
plt.tight_layout()
plt.savefig('notebooks/fig_temperature.png', bbox_inches='tight')
plt.show()
print("τ 太小→loss爆炸; τ 太大→区分度不够; 0.07 是常用值。")""")

md("""## 5. 对比学习家族对比

| 方法 | 正样本 | 负样本 | 特点 |
|------|--------|--------|------|
| **SimCLR** | batch内增强 | batch内其他 | 简单, 需大batch |
| **MoCo** | batch内增强 | 队列 | 小batch可行 |
| **CLIP** | 图文配对 | batch内其他配对 | 跨模态 |
| **NNCLR** | 最近邻 | batch内其他 | 不用增强 |
| **SupCon** | 同类标签 | 不同类 | 监督对比 |

### 演进脉络

```
InstDisc (2018)  → 个体判别 + memory bank
MoCo (2019)      → 动量编码器 + 队列
SimCLR (2020)    → 简化, 大batch
BYOL (2020)      → 不需要负样本!
DINO (2021)      → 自蒸馏, 无需负样本
```

> 💡 对比学习的后续：**不需要负样本**也能学好表示（BYOL/DINO）。
# 这是下一章的主题。""")

code("""# 对比学习表示质量评估
def evaluate_representation(model, n_classes=10, n_per_class=50):
    # 模拟: 生成各类样本, 编码, 计算类内/类间距离
    torch.manual_seed(42)
    centers = torch.randn(n_classes, 784)

    intra_distances = []  # 类内距离
    inter_distances = []  # 类间距离

    for i in range(n_classes):
        samples = centers[i:i+1] + 0.3 * torch.randn(n_per_class, 784)
        embeds = model(samples)
        # 类内距离
        for j in range(n_per_class):
            for k in range(j+1, n_per_class):
                intra_distances.append((embeds[j] - embeds[k]).norm().item())

    for i in range(n_classes):
        for j in range(i+1, n_classes):
            samples_i = centers[i:i+1] + 0.3 * torch.randn(5, 784)
            samples_j = centers[j:j+1] + 0.3 * torch.randn(5, 784)
            embeds_i = model(samples_i)
            embeds_j = model(samples_j)
            for a in range(5):
                for b in range(5):
                    inter_distances.append((embeds_i[a] - embeds_j[b]).norm().item())

    return np.mean(intra_distances), np.mean(inter_distances)

intra, inter = evaluate_representation(model)
ratio = inter / intra

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(['类内距离', '类间距离'], [intra, inter], color=['green', 'red'], alpha=0.8)
ax.set_ylabel('距离')
ax.set_title(f'对比学习表示质量 (类间/类内 = {ratio:.2f})')
plt.tight_layout()
plt.savefig('notebooks/fig_repr_quality.png', bbox_inches='tight')
plt.show()
print(f"类内距离: {intra:.3f}, 类间距离: {inter:.3f}")
print(f"比值: {ratio:.2f} (越大越好——同类近, 异类远)")""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 自监督学习动机 | ✅ |
| SimCLR 框架 | ✅ |
| MoCo 队列+动量 | ✅ |
| InfoNCE 损失 | ✅ |
| 温度参数 | ✅ |

### 核心 takeaway
> **对比学习 = 拉近正样本 + 推开负样本**。
> SimCLR 简单但需大 batch，MoCo 用队列解耦。
> InfoNCE 是互信息下界——负样本越多，表示越好。

### 🔗 下一章
**`31_non_contrastive_ssl.ipynb`** — BYOL/DINO/DINOv2 非对比自监督

---

> 💬 **板块五进行中。**""")

output_path = "notebooks/30_contrastive_learning.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")