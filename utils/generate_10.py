"""生成 10_pretraining_scaling.ipynb 的脚本"""
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
md("""# 10 — 自监督预训练 + Scaling Law

> 标注数据是瓶颈——人类标注 100 万条要花几百万美元。
> 自监督预训练用**数据自己教自己**：从海量无标注文本中学习语言规律。
> 这就是 GPT/BERT/Llama 的第一阶段——预训练。

## 本章你将掌握

1. **自监督学习**：为什么不需要标注
2. **MLM**：掩码语言建模（BERT 用的）
3. **CLM**：因果语言建模（GPT 用的）
4. **MAE**：掩码自编码器（视觉用的）
5. **Scaling Law**：参数/数据/算力的幂律关系""")

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
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
torch.manual_seed(42)
print("环境准备完成 ✅")""")

# ============================================================
md("""## 1. 自监督学习：数据自己教自己

### 1.1 监督 vs 自监督

```
监督学习:   "猫的图片" → 标签: "猫"     (需要人工标注)
自监督:     "猫的 [MASK] 很可爱" → 预测: "图片"  (从数据自己构造标签)
```

自监督的巧妙之处——**从无标注数据中自动构造训练信号**：
- **MLM**：遮住一些 token，预测被遮住的（完形填空）
- **CLM**：给定前文，预测下一个 token（接龙）
- **MAE**：遮住图片块，重建被遮住的

### 1.2 为什么自监督这么强？

| 优势 | 说明 |
|------|------|
| **数据无限** | 互联网上有无限无标注文本 |
| **信号丰富** | 每个 token 都是一个训练信号 |
| **学到表示** | 预训练学到通用表示，下游微调即可用 |

> 💡 GPT-4 预训练用了约 13T token——如果用监督学习标注这么多，成本不可想象。""")

code("""# 准备一个小语料
corpus = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat ran on the floor",
    "the dog ran on the grass",
    "the bird flew in the sky",
    "the fish swam in the sea",
    "the cat jumped on the wall",
    "the dog jumped on the bed",
]
vocab = sorted(set(' '.join(corpus).split()))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}
vocab_size = len(vocab)
print(f"语料: {len(corpus)} 句")
print(f"词表: {vocab} (大小={vocab_size})")""")

# ============================================================
md("""## 2. CLM：因果语言建模（GPT 用的）

### 2.1 CLM 的训练目标

给定前文，预测下一个 token：

$$L = -\\sum_t \\log P(x_t | x_{<t})$$

```
输入:  the  cat  sat  on  the  mat
目标:  cat  sat  on   the  mat  <EOS>
       ↑    ↑    ↑    ↑    ↑    ↑
      预测每个位置的下一个token
```

### 2.2 CLM 的特点

- **单向**：只看左边（因果掩码）
- **生成式**：训练完能直接生成文本
- **每个 token 都是训练信号**：序列长度 N → N 个训练样本""")

code("""class TinyTransformerCLM(nn.Module):
    \"\"\"简化 Transformer 用于 CLM\"\"\"
    def __init__(self, vocab_size, d_model=32, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        T = x.size(1)
        x = self.embed(x) + self.pos_embed[:, :T]
        # 因果掩码: 只看左边
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        x = self.encoder(x, mask=mask)
        return self.head(x)

# 准备 CLM 数据
def make_clm_data(corpus, word2id, max_len=10):
    sequences = []
    for sentence in corpus:
        ids = [word2id[w] for w in sentence.split()]
        if len(ids) > 1:
            sequences.append(ids)
    return sequences

clm_data = make_clm_data(corpus, word2id)
print(f"CLM 数据: {len(clm_data)} 个序列")
print(f"示例: {[id2word[i] for i in clm_data[0]]}")

# 训练 CLM
clm_model = TinyTransformerCLM(vocab_size)
optimizer = torch.optim.Adam(clm_model.parameters(), lr=1e-3)
losses_clm = []

for epoch in range(200):
    total_loss = 0
    for seq in clm_data:
        x = torch.tensor([seq[:-1]])  # 输入: 前N-1个token
        y = torch.tensor([seq[1:]])   # 目标: 后N-1个token
        logits = clm_model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses_clm.append(total_loss / len(clm_data))

print(f"CLM 训练完成, 最终损失: {losses_clm[-1]:.4f}")""")

code("""# 用 CLM 生成文本
def generate_clm(model, start_tokens, max_new=5, temperature=1.0):
    \"\"\"自回归生成\"\"\"
    tokens = start_tokens.copy()
    for _ in range(max_new):
        x = torch.tensor([tokens])
        logits = model(x)
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        tokens.append(next_id)
        if next_id >= vocab_size:
            break
    return tokens

# 从 "the cat" 开始生成
start = [word2id['the'], word2id['cat']]
generated = generate_clm(clm_model, start, max_new=4)
print(f"CLM 生成: {' '.join(id2word[i] for i in generated)}")

# 可视化训练曲线
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses_clm, 'b-', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('CLM 训练损失'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_clm_training.png', bbox_inches='tight')
plt.show()
print("CLM 训练后能生成文本——这是 GPT 的核心能力。")""")

# ============================================================
md("""## 3. MLM：掩码语言建模（BERT 用的）

### 3.1 MLM 的训练目标

随机遮住 15% 的 token，预测被遮住的：

$$L = -\\sum_{i \\in \\text{masked}} \\log P(x_i | x_{\\text{unmasked}})$$

```
原始: the cat sat on the mat
掩码: the [MASK] sat on the mat   → 预测: cat
```

### 3.2 MLM vs CLM

| | CLM (GPT) | MLM (BERT) |
|---|-----------|------------|
| **方向** | 单向（左→右） | 双向 |
| **目标** | 预测下一个 | 预测被遮住的 |
| **生成** | ✅ 能生成 | ❌ 不能直接生成 |
| **理解** | ⚠️ 较弱 | ✅ 更强 |

> 💡 MLM 能看到右边上下文 → 理解任务更强。但遮码破坏了生成能力 → BERT 不能生成文本。""")

code("""class TinyTransformerMLM(nn.Module):
    \"\"\"简化 Transformer 用于 MLM (双向)\"\"\"
    def __init__(self, vocab_size, d_model=32, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        T = x.size(1)
        x = self.embed(x) + self.pos_embed[:, :T]
        x = self.encoder(x)  # 不加因果掩码 → 双向
        return self.head(x)

def apply_mlm(seq, mask_prob=0.15, mask_id=0):
    \"\"\"随机遮码: 返回 (遮码后序列, 标签, 遮码位置)\"\"\"
    masked = seq.copy()
    labels = [-100] * len(seq)  # -100 = 不计算损失
    mask_positions = []
    for i in range(len(seq)):
        if np.random.random() < mask_prob:
            labels[i] = seq[i]
            masked[i] = mask_id  # 用0当MASK token
            mask_positions.append(i)
    return masked, labels, mask_positions

# 训练 MLM
mlm_model = TinyTransformerMLM(vocab_size)
optimizer = torch.optim.Adam(mlm_model.parameters(), lr=1e-3)
losses_mlm = []

for epoch in range(200):
    total_loss = 0
    n_batches = 0
    for seq in clm_data:
        masked, labels, positions = apply_mlm(seq)
        if not positions:
            continue
        x = torch.tensor([masked])
        y = torch.tensor([labels])
        logits = mlm_model(x)
        # 只在被遮码位置算损失
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    if n_batches > 0:
        losses_mlm.append(total_loss / n_batches)

print(f"MLM 训练完成, 最终损失: {losses_mlm[-1]:.4f}")

# 演示 MLM 预测
test_seq = [word2id[w] for w in "the cat sat on the mat".split()]
masked, labels, positions = apply_mlm(test_seq, mask_prob=0.3)
masked_words = [id2word.get(i, '[MASK]') if i != 0 else '[MASK]' for i in masked]
print(f"\\nMLM 演示:")
print(f"  掩码后: {' '.join(masked_words)}")
with torch.no_grad():
    logits = mlm_model(torch.tensor([masked]))
    for pos in positions:
        pred = logits[0, pos].argmax().item()
        print(f"  位置 {pos}: 预测 '{id2word[pred]}', 真实 '{id2word[labels[pos]]}'")""")

# ============================================================
md("""## 4. MAE：掩码自编码器（视觉用的）

### 4.1 MAE 的思想

MAE 把 MLM 的思想搬到视觉——**遮住图片的 75%，重建被遮住的像素**：

```
原始图片:          遮住75%:          重建:
┌─────────┐       ┌─────────┐       ┌─────────┐
│ ▓ ▓ ▓ ▓ │       │ ▓ ▓ ▓ ▓ │       │ ▓ ▓ ▓ ▓ │
│ ▓ ▓ ▓ ▓ │  →    │ ▓ ▓ ▓ ▓ │  →    │ ▓ ▓ ▓ ▓ │
│ ▓ ▓ ▓ ▓ │       │ ▓ ▓ ▓ ▓ │       │ ▓ ▓ ▓ ▓ │
│ ▓ ▓ ▓ ▓ │       │ ▓ ▓ ▓ ▓ │       │ ▓ ▓ ▓ ▓ │
└─────────┘       └─────────┘       └─────────┘
```

### 4.2 MAE 的两个特点

1. **遮码比例高**（75%）：比 MLM 的 15% 高得多——图片信息冗余度高
2. **编码器只看可见块**：被遮住的不进 encoder → 大幅节省计算

> 💡 MAE 的 75% 遮码让 encoder 只处理 25% 的 patch → 计算量减少 4x。""")

code("""# 简化 MAE 演示: 用 MNIST 数字
from sklearn.datasets import load_digits

digits = load_digits()
image = digits.images[0].astype(np.float32) / 16.0  # 8x8 归一化

# 遮码 75%
mask_ratio = 0.75
n_pixels = image.size
n_mask = int(n_pixels * mask_ratio)
mask = np.zeros(n_pixels, dtype=bool)
mask[np.random.choice(n_pixels, n_mask, replace=False)] = True
mask = mask.reshape(image.shape)

visible = image.copy()
visible[mask] = 0  # 遮住的部分设为0

# 简化重建: 用可见像素的平均值填充 (真实MAE用Transformer重建)
reconstructed = visible.copy()
reconstructed[mask] = visible[~mask].mean()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray'); axes[0].set_title('原始图片')
axes[1].imshow(visible, cmap='gray'); axes[1].set_title(f'遮住 {mask_ratio:.0%}')
axes[2].imshow(reconstructed, cmap='gray'); axes[2].set_title('重建 (简化)')
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('notebooks/fig_mae_demo.png', bbox_inches='tight')
plt.show()
print(f"MAE: 遮住 {mask_ratio:.0%} 像素, 只用 {1-mask_ratio:.0%} 可见像素重建。")
print("真实 MAE 用 ViT encoder 处理可见 patch, 轻轻量 decoder 重建。")""")

# ============================================================
md("""## 5. Scaling Law：规模的力量

### 5.1 Kaplan 撑律

OpenAI 2020 年发现损失随参数量 $N$、数据量 $D$、算力 $C$ 呈**幂律**下降：

$$L(N) = \\left(\\frac{N_c}{N}\\right)^{\\alpha_N}, \\quad L(D) = \\left(\\frac{D_c}{D}\\right)^{\\alpha_D}, \\quad L(C) = \\left(\\frac{C_c}{C}\\right)^{\\alpha_C}$$

```
损失
 │
 │ ╲
 │  ╲  L ∝ N^(-0.076)
 │   ╲
 │    ╲________________
 │
 └──────────────────── 参数量 N (对数尺度)
```

### 5.2 Chinchilla 修正

DeepMind 2022 年发现 **参数和数据要同步增长**：

$$D_{opt} \\approx 20 \\cdot N$$

| 模型 | 参数 | 数据 | 是否最优 |
|------|------|------|---------|
| GPT-3 | 175B | 300B | ❌ 欠训练 |
| Chinchilla | 70B | 1.4T | ✅ 最优 |
| Llama 2 | 70B | 2T | ✅ 过训练一点 |

> 💡 Chinchilla 定律：**不要只堆参数，数据要跟上**。70B 模型需要约 1.4T token 才最优。""")

code("""# 模拟 Scaling Law
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 损失 vs 参数量
N = np.logspace(8, 12, 100)  # 100M to 1T
# Kaplan: L(N) = (Nc/N)^alpha
Nc = 8.8e8  # 临界参数量
alpha_N = 0.076
L_N = (Nc / N) ** alpha_N + 1.5  # +不可约损失
axes[0].plot(N, L_N, 'b-', linewidth=2.5)
axes[0].set_xscale('log')
axes[0].set_xlabel('参数量 N')
axes[0].set_ylabel('损失 L')
axes[0].set_title('Scaling Law: 损失 vs 参数量')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(1.5, color='r', linestyle='--', alpha=0.5, label='不可约损失')
axes[0].legend()

# 2. 损失 vs 数据量
D = np.logspace(9, 13, 100)  # 1B to 10T
Dc = 5.4e9
alpha_D = 0.095
L_D = (Dc / D) ** alpha_D + 1.5
axes[1].plot(D, L_D, 'g-', linewidth=2.5)
axes[1].set_xscale('log')
axes[1].set_xlabel('数据量 D (tokens)')
axes[1].set_ylabel('损失 L')
axes[1].set_title('Scaling Law: 损失 vs 数据量')
axes[1].grid(True, alpha=0.3)

# 3. Chinchilla: 最优数据/参数比
N_range = np.logspace(8, 11.5, 50)
D_opt = 20 * N_range  # Chinchilla: D ≈ 20N
D_gpt3 = np.full_like(N_range, 300e9)  # GPT-3: 固定300B
axes[2].plot(N_range, D_opt, 'b-', linewidth=2.5, label='Chinchilla 最优 (D≈20N)')
axes[2].plot(N_range, D_gpt3, 'r--', linewidth=2, label='GPT-3 (D=300B 固定)')
axes[2].set_xscale('log'); axes[2].set_yscale('log')
axes[2].set_xlabel('参数量 N')
axes[2].set_ylabel('最优数据量 D')
axes[2].set_title('Chinchilla: 参数与数据要同步增长')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_scaling_law.png', bbox_inches='tight')
plt.show()
print("Scaling Law: 损失随参数/数据/算力幂律下降——但最终会触底(不可约损失)。")""")

code("""# 模拟不同规模模型的训练
print("模拟不同规模模型 (按 Chinchilla 定律):")
print(f"{'模型':12s} {'参数':>10s} {'数据':>10s} {'理论损失':>10s} {'是否最优':>10s}")
print("-" * 60)

models = [
    ("小型", 100e6, 2e9),
    ("中型", 1e9, 20e9),
    ("大型", 10e9, 200e9),
    ("GPT-3", 175e9, 300e9),
    ("Chinchilla", 70e9, 1.4e12),
    ("Llama-2-70B", 70e9, 2e12),
]

for name, N, D in models:
    L = (Nc / N) ** alpha_N + (Dc / D) ** alpha_D + 1.5
    ratio = D / N
    optimal = "✅" if 15 < ratio < 25 else ("⚠️ 欠训练" if ratio < 15 else "🟡 过训练")
    print(f"{name:12s} {N/1e9:>9.1f}B {D/1e12:>9.2f}T {L:>10.4f} {optimal:>10s}")

print("\\nChinchilla 定律: D ≈ 20N 时最优。GPT-3 参数多但数据不够 → 欠训练。")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 自监督学习：从数据自己构造标签 | ✅ |
| CLM 因果语言建模（GPT） | ✅ |
| MLM 掩码语言建模（BERT） | ✅ |
| MAE 掩码自编码器（视觉） | ✅ |
| Scaling Law 幂律关系 | ✅ |
| Chinchilla 定律：D ≈ 20N | ✅ |

### 核心 takeaway

> **自监督预训练 + 规模 = 现代AI的基础**。
> CLM 让模型能生成，MLM 让模型能理解，Scaling Law 告诉我们怎么高效扩展。
> Chinchilla 定律：**不要只堆参数，数据要同步跟上**。

### 🔗 下一章预告

**`11_sft_peft.ipynb`** — SFT + LoRA/QLoRA + DoRA/PiSSA（微调技术）

---

> 💬 **写在最后**：预训练是"通识教育"，微调是"专业训练"。
> Scaling Law 是规模化的指南针——它告诉我们投资参数还是数据。""")

# ============================================================
output_path = "notebooks/10_pretraining_scaling.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")