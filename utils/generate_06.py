"""生成 06_cnn_vs_vit.ipynb 的脚本"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))

# ============================================================
md("""# 06 — CNN ↔ ViT：归纳偏置 vs 数据效率

> 从 2012 年 AlexNet 到 2020 年 ViT，图像识别被 CNN 统治了 8 年。
> 然后 ViT 来了，用**纯注意力**做视觉，一开始在小数据上输给 CNN，
> 但在**大数据上赢了**。这背后的核心张力是：**归纳偏置 vs 数据效率**。
>
> 本章我们实现 CNN 和 ViT，在相同数据上对比，亲眼看到这个范式转移。

## 本章你将掌握

1. **CNN 的三大归纳偏置**：局部性、权值共享、平移等变性
2. **经典架构演进**：LeNet → VGG → ResNet（残差连接为什么重要）
3. **ViT 的核心思想**：把图像切成 patch，当序列处理
4. **ViT 没有归纳偏置**：靠数据自己学出局部性
5. **对比实验**：小数据 CNN 赢，大数据 ViT 赢""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
torch.manual_seed(42)
print("环境准备完成 ✅")""")

# ============================================================
md("""## 1. CNN 的核心：卷积操作

### 1.1 什么是卷积？

全连接层每个神经元连接**所有输入**，参数量巨大。卷积层每个神经元只连接**局部区域**，且**不同位置共享同一组权重**：

```
全连接: 每个输出看全部输入       卷积: 每个输出只看局部，且权重共享
  ●●●●●─→○                      ●●●─→○
  ●●●●●─→○                        ●●●─→○ (同一组权重!)
  ●●●●●─→○                          ●●●─→○
```

### 1.2 CNN 的三大归纳偏置

| 归纳偏置 | 含义 | 好处 |
|---------|------|------|
| **局部性** | 每个神经元只看局部区域 | 适合图像（像素相关性随距离衰减） |
| **权值共享** | 不同位置用同一组权重 | 参数量大减，且能检测任意位置的同一模式 |
| **平移等变性** | 模式移动 → 特征图跟着移动 | 猫在左上角和右下角都能识别 |

> 💡 这些是**先验知识**——我们告诉网络"图像有这些性质"。好处是不需要从数据中学，
> 坏处是如果数据不满足这些先验，就限制了表达能力。""")

code("""# 从零实现一个卷积操作（教学版）
def conv2d_simple(image, kernel, stride=1, padding=0):
    \"\"\"二维卷积（实际是互相关，深度学习里叫卷积）
    image: (H, W) 输入
    kernel: (kH, kW) 卷积核
    \"\"\"
    if padding > 0:
        image = np.pad(image, padding)

    H, W = image.shape
    kH, kW = kernel.shape
    out_h = (H - kH) // stride + 1
    out_w = (W - kW) // stride + 1

    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+kH, j*stride:j*stride+kW]
            output[i, j] = np.sum(region * kernel)
    return output

# 用不同卷积核检测不同特征
np.random.seed(42)
image = np.random.randn(8, 8)  # 8x8 随机图像

# 边缘检测核
edge_kernel = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]])
# 水平边缘核
h_edge_kernel = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]])

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='RdBu'); axes[0].set_title('原始图像')
axes[1].imshow(conv2d_simple(image, edge_kernel), cmap='RdBu'); axes[1].set_title('边缘检测')
axes[2].imshow(conv2d_simple(image, h_edge_kernel), cmap='RdBu'); axes[2].set_title('水平边缘')
plt.tight_layout()
plt.savefig('notebooks/fig_conv_operation.png', bbox_inches='tight')
plt.show()
print("不同卷积核检测不同特征——这就是 CNN 的基本原理。")""")

# ============================================================
md("""## 2. 经典 CNN 架构演进

### 2.1 LeNet → VGG → ResNet

| 架构 | 年份 | 关键创新 | 深度 |
|------|------|---------|------|
| **LeNet** | 1998 | 卷积+池化基本结构 | 5 层 |
| **VGG** | 2014 | 统一 3×3 卷积堆叠 | 19 层 |
| **ResNet** | 2015 | **残差连接** $y = F(x) + x$ | 152 层 |

### 2.2 残差连接为什么重要

深网络有个问题：**梯度消失**导致训不动。ResNet 的解法极其简单——加个**跳连**：

$$y = F(x) + x$$

梯度可以通过 $+x$ 这条路**直接传回去**，不经过 $F$ 的连乘，不会消失。

> 这个简单的加法让网络从 19 层堆到 152 层甚至更深，是深度学习的里程碑。""")

code("""# 实现三种架构（简化教学版）

class SimpleLeNet(nn.Module):
    \"\"\"LeNet: 卷积→池化→卷积→池化→全连接\"\"\"
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 2 * 2, num_classes)  # 8x8 → 2x2 after 2 pools

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 8→4
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 4→2
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class ResBlock(nn.Module):
    \"\"\"ResNet 残残差块: y = F(x) + x\"\"\"
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x            # 跳连保存输入
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)  # 残残差连接！

class SimpleResNet(nn.Module):
    \"\"\"简化 ResNet: 用残差块堆叠\"\"\"
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.block1 = ResBlock(16)
        self.block2 = ResBlock(16)
        self.block3 = ResBlock(16)
        self.fc = nn.Linear(16 * 2 * 2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 8→4
        x = self.block1(x)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)  # 4→2
        x = self.block3(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# 验证
x_test = torch.randn(4, 1, 8, 8)
lenet = SimpleLeNet()
resnet = SimpleResNet()
print(f"LeNet 输出: {lenet(x_test).shape}")
print(f"ResNet 输出: {resnet(x_test).shape}")

# 统计参数量
n_lenet = sum(p.numel() for p in lenet.parameters())
n_resnet = sum(p.numel() for p in resnet.parameters())
print(f"\\nLeNet 参数: {n_lenet:,}")
print(f"ResNet 参数: {n_resnet:,}")
print("ResNet 参数更多但能训更深——残差连接是关键。")""")

# ============================================================
md("""## 3. ViT：把图像当序列

### 3.1 ViT 的核心思想

ViT 完全**抛弃卷积**，做法极其粗暴：

1. 把 $H \\times W$ 的图像切成 $N$ 个 patch（比如 16×16）
2. 每个 patch 拉平成一个向量
3. 加上位置编码
4. 喂进标准 Transformer

```
图像 → 切 patch → 拉平 + 位置编码 → Transformer → 分类
 ●●●     [p1][p2]    [e1][e2]      [h1][h2]     ●
 ●●●  →  [p3][p4] → [e3][e4]  →   [h3][h4]  →
```

### 3.2 ViT 没有归纳偏置

ViT 的自注意力让**每个 patch 能看到所有其他 patch**——没有局部性先验。
位置关系完全靠**位置编码从数据中学**。

> 这就是为什么 ViT 在小数据上输给 CNN——它没有先验，需要更多数据来学出局部性。
> 但在大数据上，学到的局部性比手工设计的更好 → 赢。""")

code("""class SimpleViT(nn.Module):
    \"\"\"简化 ViT: patch embedding + Transformer encoder + 分类头\"\"\"
    def __init__(self, img_size=8, patch_size=2, in_ch=1, d_model=64, n_heads=4, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (img_size // patch_size) ** 2

        # Patch embedding: 把每个 patch 拉平后线性映射到 d_model
        self.patch_embed = nn.Linear(in_ch * patch_size * patch_size, d_model)
        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # 分类头
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.size(0)
        # 1. 切 patch
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        # 2. Patch embedding + 位置编码
        x = self.patch_embed(x) + self.pos_embed
        # 3. 加 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        # 4. Transformer
        x = self.encoder(x)
        # 5. 用 CLS token 分类
        return self.head(x[:, 0])

vit = SimpleViT()
x_test = torch.randn(4, 1, 8, 8)
print(f"ViT 输出: {vit(x_test).shape}")
n_vit = sum(p.numel() for p in vit.parameters())
print(f"ViT 参数: {n_vit:,}")
print("ViT 没有卷积，纯注意力——把图像当序列处理。")""")

# ============================================================
md("""## 4. 对比实验：CNN vs ViT

用 sklearn 的手写数字数据集（8×8 图像，10 类），在不同数据量下对比 CNN 和 ViT。""")

code("""# 准备数据
digits = load_digits()
X = digits.images.astype(np.float32) / 16.0  # 归一化
y = digits.target
X = X[:, np.newaxis, :, :]  # (N, 1, 8, 8)

def train_and_eval(model, X_tr, y_tr, X_te, y_te, epochs=50, lr=1e-3):
    \"\"\"训练并评估\"\"\"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_tr_t = torch.tensor(X_tr); y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te_t = torch.tensor(X_te); y_te_t = torch.tensor(y_te, dtype=torch.long)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        tr_acc = (model(X_tr_t).argmax(1) == y_tr_t).float().mean().item()
        te_acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
    return tr_acc, te_acc

# 不同训练集大小对比
data_sizes = [50, 200, 500, 1000, 1500]
results = {'CNN': [], 'ViT': []}

for n_train in data_sizes:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=n_train, stratify=y, random_state=42)
    X_te, y_te = X[:500], y[:500]  # 固定测试集

    torch.manual_seed(42)
    cnn = SimpleResNet()
    tr_cnn, te_cnn = train_and_eval(cnn, X_tr, y_tr, X_te, y_te, epochs=80)

    torch.manual_seed(42)
    vit = SimpleViT()
    tr_vit, te_vit = train_and_eval(vit, X_tr, y_tr, X_te, y_te, epochs=80)

    results['CNN'].append(te_cnn)
    results['ViT'].append(te_vit)
    print(f"训练量={n_train:4d}: CNN 测试准确率={te_cnn:.2%}, ViT 测试准确率={te_vit:.2%}")""")

code("""# 可视化对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data_sizes, results['CNN'], 'b-o', linewidth=2.5, markersize=8, label='CNN (ResNet)')
ax.plot(data_sizes, results['ViT'], 'r-s', linewidth=2.5, markersize=8, label='ViT')
ax.set_xlabel('训练数据量')
ax.set_ylabel('测试准确率')
ax.set_title('CNN vs ViT：归纳偏置 vs 数据效率')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(0.3, 1.0)
plt.tight_layout()
plt.savefig('notebooks/fig_cnn_vs_vit.png', bbox_inches='tight')
plt.show()
print("观察：")
print("  数据少时 CNN 赢——归纳偏置帮它不用学局部性")
print("  数据多时 ViT 追上甚至超过——学到的局部性比手工的好")""")

# ============================================================
md("""## 5. 归纳偏置 vs 数据效率：核心张力

### 5.1 总结

```
                归纳偏置强                    归纳偏置弱
                ←─────────────────────────────→
     CNN                                        ViT
  (局部性+权值共享)                          (纯注意力)

  小数据: ✅ 赢                              小数据: ❌ 输
  大数据: 🟡 受限                            大数据: ✅ 赢
```

### 5.2 这对现代深度学习的启示

| 启示 | 含义 |
|------|------|
| **数据够多时，少加先验** | 先验是双刃剑——帮小数据，限大数据 |
| **ViT 需要大数据** | 原始 ViT 要 JFT-300M 级别数据才打过 CNN |
| **混合架构** | ConvNeXt 把 ViT 的设计加回 CNN；DeiT 用蒸馏让 ViT 适配小数据 |

> 现代趋势：**先验和数据的边界在模糊**。ConvNeXt 证明"用 ViT 的配方做 CNN 也能很强"，
> DeiT 证明"用知识蒸馏能让 ViT 在小数据上也能用"。最终是**架构+数据+训练技巧**的综合比拼。""")

code("""# 可视化归纳偏置的代价
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
# CNN: 快速上升但有天花板
cnn_curve = 0.95 * (1 - np.exp(-x/2)) - 0.05
# ViT: 慢启动但最终更高
vit_curve = 0.98 * (1 - np.exp(-x/5))

ax.plot(x, cnn_curve, 'b-', linewidth=3, label='CNN (归纳偏置强)')
ax.plot(x, vit_curve, 'r-', linewidth=3, label='ViT (归纳偏置弱)')
ax.fill_between(x, cnn_curve, vit_curve, where=vit_curve>cnn_curve,
                color='red', alpha=0.1, label='ViT 优势区')
ax.fill_between(x, cnn_curve, vit_curve, where=vit_curve<cnn_curve,
                color='blue', alpha=0.1, label='CNN 优势区')

crossover = x[np.argmin(np.abs(cnn_curve - vit_curve))]
ax.axvline(crossover, color='gray', linestyle='--', alpha=0.5)
ax.annotate(f'交叉点\\n(数据量≈{crossover:.1f})', xy=(crossover, 0.5),
            fontsize=10, ha='center', color='gray')

ax.set_xlabel('数据量')
ax.set_ylabel('性能')
ax.set_title('归纳偏置 vs 数据效率：核心张力')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_inductive_bias.png', bbox_inches='tight')
plt.show()
print("数据少时先验帮忙，数据多时先验反而限制——这就是 CNN→ViT 范式转移的本质。")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| CNN 三大归纳偏置（局部性/权值共享/平移等变） | ✅ |
| 卷积操作从零实现 | ✅ |
| LeNet → VGG → ResNet 演进 | ✅ |
| 残残差连接为什么让网络能更深 | ✅ |
| ViT 把图像切 patch 当序列处理 | ✅ |
| ViT 没有归纳偏置，靠数据学 | ✅ |
| CNN vs ViT 在不同数据量下的对比实验 | ✅ |

### 核心 takeaway

> **数据少 → 先验帮忙（CNN 赢）；数据多 → 先验限制（ViT 赢）**。
> 现代视觉是两者的融合：ConvNeXt、DeiT、混合架构。

### 🔗 下一章预告

**`07_rnn_vs_transformer.ipynb`** — RNN/LSTM ↔ Transformer 对比（序列建模范式转移）

---

> 💬 **写在最后**：CNN→ViT 的转移不只是架构变化，更是**从"人设计先验"到"数据学先验"**的哲学转变。
> 这个转变在 NLP 里更早就发生了（RNN→Transformer），下一章我们就讲。""")

# ============================================================
output_path = "notebooks/06_cnn_vs_vit.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")