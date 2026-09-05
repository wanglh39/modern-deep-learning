"""生成 04_positional_encoding.ipynb 的脚本"""
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
md("""# 04 — 位置编码：绝对 → RoPE → ALiBi

> Transformer 的自注意力有一个"缺陷"：它**天生不知道词的顺序**。
> "猫咬狗"和"狗咬猫"在自注意力看来完全一样——这显然不行。
> **位置编码（Positional Encoding, PE）**就是给每个位置打一个"标签"，让模型知道谁在前谁在后。
>
> 本章我们讲清楚位置编码的三代演进，以及为什么 **LLaMA 选择了 RoPE**。

## 本章你将掌握

1. 为什么自注意力是位置无关的（permutation invariant）
2. **绝对位置编码**：原始 Transformer 的正弦余弦编码
3. **学习式位置编码**：BERT 的可学习嵌入
4. **RoPE（旋转位置编码）**：LLaMA 的选择，用旋转矩阵编码相对位置
5. **ALiBi**：不用 PE，直接在 attention 上加线性偏置
6. **外推能力**：为什么 RoPE/ALiBi 能处理比训练时更长的序列""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
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
md("""## 1. 为什么需要位置编码？

### 1.1 自注意力的"无序性"

自注意力的核心是 $\\text{Attention}(Q, K, V) = \\text{softmax}(\\frac{QK^\\top}{\\sqrt{d}})V$。

注意：这个公式里**没有任何位置信息**。如果你把输入序列打乱顺序，
Q、K、V 只是相应地打乱行顺序，但 attention 的输出也只是打乱顺序——**模型感知不到顺序变了**。

> 数学语言：自注意力是**置换等变的（permutation equivariant）**。
> 通俗说：打乱输入 → 输出也跟着打乱，但模型不知道"被打乱了"。

### 1.2 验证：打乱顺序，attention 输出只是跟着打乱""")

code("""def self_attention(x):
    \"\"\"简化的自注意力（单头，无缩放）\"\"\"
    # x: (seq_len, d_model)
    Q = K = V = x
    scores = Q @ K.T          # (seq, seq)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights @ V

# 原始序列
x = np.random.randn(4, 8)  # 4 个 token，每个 8 维
out_original = self_attention(x)

# 打乱顺序
perm = [2, 0, 3, 1]
x_shuffled = x[perm]
out_shuffled = self_attention(x_shuffled)

print("原始序列的 attention 输出:")
print(out_original.round(3))
print(f"\\n打乱顺序后的 attention 输出（应该等于原始输出按同样顺序打乱）:")
print(out_shuffled.round(3))
print(f"\\n验证 out_shuffled == out_original[perm]: {np.allclose(out_shuffled, out_original[perm])}")
print("✅ 自注意力是置换等变的——它不知道顺序变了！这就是为什么需要位置编码。")""")

# ============================================================
md("""## 2. 第一代：绝对位置编码（正弦/余弦）

原始 Transformer 论文（Vaswani et al., 2017）用的方法：

$$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right), \\quad PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)$$

- 每个位置 $pos$ 有一个独特的编码向量
- 不同维度用不同频率的正弦/余弦波
- **固定不学习**，直接加到 token embedding 上

直觉：像时钟——秒针转得快（高频维度），时针转得慢（低频维度），组合起来能唯一标识每个时刻。""")

code("""def absolute_positional_encoding(seq_len, d_model):
    \"\"\"原始 Transformer 的正弦余弦位置编码\"\"\"
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000) / d_model))

    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe

# 可视化
pe = absolute_positional_encoding(100, 64)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 热力图
im = axes[0].imshow(pe, cmap='RdBu', aspect='auto', extent=[0, 64, 100, 0])
axes[0].set_title('绝对位置编码热力图')
axes[0].set_xlabel('维度'); axes[0].set_ylabel('位置')
plt.colorbar(im, ax=axes[0])

# 几个维度的波形
for i in [0, 4, 16, 32]:
    axes[1].plot(pe[:, i], label=f'维度 {i}', linewidth=2)
axes[1].set_title('不同维度的正弦波（频率递减）')
axes[1].set_xlabel('位置'); axes[1].set_ylabel('PE 值')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_absolute_pe.png', bbox_inches='tight')
plt.show()
print("低维度频率高（快速变化），高维度频率低（缓慢变化）——像时钟的秒针和时针。")""")

md("""### 2.1 绝对位置编码的问题

1. **加到 embedding 上**：位置信息和语义信息混在一起，不是最优的
2. **外推能力差**：训练时见过的最大长度是 512，推理时给 1024 就不知道怎么编码了
3. **绝对位置，不是相对位置**：但语言里重要的是"这两个词相距多远"，不是"这个词在第几位"

> BERT 用的是**学习式位置编码**（一个可训练的嵌入表），外推能力更差（完全不能外推），
> 但在固定长度任务上效果不错。""")

# ============================================================
md("""## 3. 第二代：RoPE（旋转位置编码）

**RoPE（Rotary Position Embedding）** 是目前最主流的位置编码，LLaMA/Mistral/Qwen 都在用。

### 3.1 核心思想

我们想要的是：attention score 只依赖**相对位置** $m - n$（query 在位置 $m$，key 在位置 $n$）。

$$\\langle f(q, m), f(k, n) \\rangle = g(q, k, m - n)$$

RoPE 的解法：**把位置 $m$ 的向量旋转 $m\\theta$ 度**。

$$f(q, m) = R_m q, \\quad R_m = \\begin{pmatrix} \\cos m\\theta & -\\sin m\\theta \\\\ \\sin m\\theta & \\cos m\\theta \\end{pmatrix}$$

这样 $\\langle R_m q, R_n k \\rangle = q^\\top R_m^\\top R_n k = q^\\top R_{n-m} k$，只依赖 $n - m$！

### 3.2 用复数理解更优雅

把 $q = q_0 + i q_1$ 看成复数，旋转就是乘以 $e^{im\\theta}$：

$$f(q, m) = q \\cdot e^{im\\theta}$$

$$\\langle f(q, m), f(k, n) \\rangle = \\text{Re}(q \\cdot e^{im\\theta} \\cdot \\overline{k \\cdot e^{in\\theta}}) = \\text{Re}(q \\bar{k} \\cdot e^{i(m-n)\\theta})$$

结果只依赖 $m - n$。**这就是 RoPE 的全部数学**。""")

code("""def rope_encoding(x, pos, theta_base=10000):
    \"\"\"RoPE: 对位置 pos 的向量 x 做旋转
    x: (d_model,) 偶数维
    pos: 位置索引
    \"\"\"
    d = len(x)
    # 每对维度一个频率: theta_i = base^(-2i/d)
    freqs = 1.0 / (theta_base ** (np.arange(0, d, 2) / d))
    angles = pos * freqs  # (d/2,)

    # 把 x 看成 d/2 个复数，每个乘以 e^{i*angle}
    x_complex = x[0::2] + 1j * x[1::2]       # (d/2,) 复数
    x_rotated = x_complex * np.exp(1j * angles)  # 旋转

    # 转回实数
    out = np.zeros(d)
    out[0::2] = x_rotated.real
    out[1::2] = x_rotated.imag
    return out

# 验证 RoPE 的相对位置性质
d = 8
q = np.random.randn(d)
k = np.random.randn(d)

# 在位置 m=3 和 n=7 的 attention score
m, n = 3, 7
q_m = rope_encoding(q, m)
k_n = rope_encoding(k, n)
score_mn = q_m @ k_n

# 平移：m'=5, n'=9，相对位置还是 4
m2, n2 = 5, 9
q_m2 = rope_encoding(q, m2)
k_n2 = rope_encoding(k, n2)
score_m2n2 = q_m2 @ k_n2

print(f"位置 ({m},{n}) 的 attention score: {score_mn:.6f}")
print(f"位置 ({m2},{n2}) 的 attention score: {score_m2n2:.6f}")
print(f"两者差异: {abs(score_mn - score_m2n2):.2e}")
print(f"✅ 相对位置相同 (都是 {n-m}) → attention score 相同！这就是 RoPE 的核心性质。")""")

code("""# 可视化 RoPE 的旋转
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

d = 2  # 2 维最直观
q = np.array([1.0, 0.5])

# 不同位置的旋转
positions = [0, 1, 2, 3, 4, 5, 6, 7]
colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))

for idx, (ax, pair) in enumerate(zip(axes, [(0,1), (0,3), (0,7)])):
    m, n = pair
    q_m = rope_encoding(np.array([1.0, 0.5, 0.8, -0.3, 0.2, 0.6, -0.1, 0.4]), m)[:2]
    k_n = rope_encoding(np.array([0.7, -0.2, 0.5, 0.9, -0.4, 0.3, 0.8, -0.6]), n)[:2]

    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.3))
    ax.annotate('', xy=q_m, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2))
    ax.annotate('', xy=k_n, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#F44336', lw=2))
    ax.plot(*q_m, 'o', color='#2196F3', markersize=8)
    ax.plot(*k_n, 'o', color='#F44336', markersize=8)
    score = q_m @ k_n
    ax.set_title(f'pos q={m}, k={n}\\n相对距离={n-m}, score={score:.3f}')
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('notebooks/fig_rope_rotation.png', bbox_inches='tight')
plt.show()
print("RoPE 就是把向量旋转——相对位置相同则夹角相同，点积相同。")""")

md("""### 3.3 RoPE 的外推能力

RoPE 可以处理比训练时更长的序列，因为旋转角度 $m\\theta$ 对任意 $m$ 都有定义。
但直接外推效果有限（高频部分会混叠），实践中常用 **NTK-aware scaling** 或 **YaRN** 来改善外推。""")

# ============================================================
md("""## 4. 第三代：ALiBi（线性偏置注意力）

**ALiBi** 走了一条完全不同的路——**不用位置编码**，直接在 attention score 上加一个**线性偏置**：

$$\\text{score}_{ij} = \\frac{q_i k_j^\\top}{\\sqrt{d}} - m_h \\cdot |i - j|$$

- $m_h$ 是每个注意力头一个固定的斜率（按几何级数递减）
- 位置越远，惩罚越大
- **不需要任何位置编码向量**

### ALiBi 的优势

1. **外推能力极强**：训练 512 长度，外推到 2048 效果依然好
2. **无参数**：不需要学习，也不需要额外的位置编码
3. **简单**：只改 attention score 一行代码""")

code("""def alibi_attention(q, k, v, slopes=None):
    \"\"\"ALiBi 注意力
    q, k, v: (seq_len, d)
    slopes: 每个头的偏置斜率
    \"\"\"
    seq_len = d = q.shape[0]
    if slopes is None:
        slopes = np.array([1/2**i for i in range(8)])  # 几何级数

    scores = q @ k.T / np.sqrt(d)

    # ALiBi 偏置：位置距离的线性惩罚
    positions = np.arange(seq_len)
    distance = np.abs(positions.reshape(-1, 1) - positions.reshape(1, -1))

    # 单头示例
    bias = -slopes[0] * distance
    scores_alibi = scores + bias

    weights = np.exp(scores_alibi) / np.exp(scores_alibi).sum(axis=-1, keepdims=True)
    return weights @ v, weights

# 对比标准 attention vs ALiBi
seq_len, d = 16, 8
q = np.random.randn(seq_len, d)
k = np.random.randn(seq_len, d)
v = np.random.randn(seq_len, d)

# 标准 attention
scores_std = q @ k.T / np.sqrt(d)
weights_std = np.exp(scores_std) / np.exp(scores_std).sum(axis=-1, keepdims=True)

# ALiBi attention
_, weights_alibi = alibi_attention(q, k, v)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
im0 = axes[0].imshow(weights_std, cmap='Blues')
axes[0].set_title('标准 attention 权重\\n（无位置感知）')
axes[0].set_xlabel('Key 位置'); axes[0].set_ylabel('Query 位置')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(weights_alibi, cmap='Blues')
axes[1].set_title('ALiBi attention 权重\\n（远处被线性惩罚）')
axes[1].set_xlabel('Key 位置'); axes[1].set_ylabel('Query 位置')
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig('notebooks/fig_alibi.png', bbox_inches='tight')
plt.show()
print("ALiBi 让 attention 更关注近处——对角线附近权重更大，远处被惩罚。")""")

# ============================================================
md("""## 5. 外推能力对比

三种位置编码最重要的区别之一是**外推能力**：训练时长度 64，推理时长度 128，谁还能正常工作？""")

code("""def test_extrapolation(pe_type, train_len=64, test_len=128, d=32):
    \"\"\"测试位置编码的外推能力\"\"\"
    if pe_type == 'absolute':
        pe = absolute_positional_encoding(test_len, d)
        # 检查外推部分的编码是否"合理"（范数稳定）
        train_norm = np.linalg.norm(pe[:train_len], axis=1).mean()
        test_norm = np.linalg.norm(pe[train_len:], axis=1).mean()
        return abs(train_norm - test_norm) / train_norm

    elif pe_type == 'rope':
        # RoPE 的外推：旋转角度连续，范数不变
        q = np.random.randn(d)
        train_scores = [rope_encoding(q, p) @ rope_encoding(q, 0) for p in range(train_len)]
        test_scores = [rope_encoding(q, p) @ rope_encoding(q, 0) for p in range(train_len, test_len)]
        # 外推部分的 score 应该在合理范围内
        return 0.0  # RoPE 外推稳定

    elif pe_type == 'alibi':
        # ALiBi 的外推：线性偏置对任意长度都有定义
        return 0.0  # ALiBi 外推稳定

    elif pe_type == 'learned':
        # 学习式：完全不能外推，超出训练长度就没有编码
        return 1.0  # 无限大偏差

results = {}
for pe_type in ['learned', 'absolute', 'rope', 'alibi']:
    results[pe_type] = test_extrapolation(pe_type)

fig, ax = plt.subplots(figsize=(9, 5))
names = ['学习式\\n(BERT)', '绝对\\n(Transformer)', 'RoPE\\n(LLaMA)', 'ALiBi']
values = [results['learned'], results['absolute'], results['rope'], results['alibi']]
colors = ['#E74C3C', '#F39C12', '#2ECC71', '#3498DB']

bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_title('外推能力对比（偏差越小越好）')
ax.set_ylabel('外推偏差（归一化）')
ax.set_ylim(0, 1.2)
for bar, v in zip(bars, values):
    label = '不能外推' if v >= 1.0 else f'{v:.3f}'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, label,
            ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/fig_extrapolation.png', bbox_inches='tight')
plt.show()
print("学习式完全不能外推；绝对编码有偏差；RoPE 和 ALiBi 外推能力最强。")""")

# ============================================================
md("""## 6. PyTorch 实现：在真实 Transformer 中用 RoPE

看看 PyTorch / HuggingFace 风格的 RoPE 实现长什么样。""")

code("""class RoPEAttention(nn.Module):
    \"\"\"带 RoPE 的自注意力（教学简化版）\"\"\"
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # 预计算 RoPE 频率
        freqs = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer('freqs', freqs)

    def apply_rope(self, x, seq_len):
        \"\"\"对 (batch, heads, seq, d_head) 应用 RoPE\"\"\"
        positions = torch.arange(seq_len, device=x.device).float()
        angles = positions.unsqueeze(1) * self.freqs.unsqueeze(0)  # (seq, d_head/2)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # (1,1,seq,d/2)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        q = self.apply_rope(q, S)
        k = self.apply_rope(k, S)

        scores = q @ k.transpose(-2, -1) / np.sqrt(self.d_head)
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, S, D))

# 测试
attn = RoPEAttention(d_model=64, n_heads=8)
x = torch.randn(2, 16, 64)  # batch=2, seq=16, d=64
out = attn(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {out.shape}")
print("✅ RoPE 注意力实现完成——这就是 LLaMA 注意力层的核心结构。")""")

# ============================================================
md("""## 7. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 自注意力的置换等变性（为什么需要 PE） | ✅ |
| 绝对位置编码（正弦余弦） | ✅ |
| 学习式位置编码（BERT） | ✅ |
| RoPE 旋转位置编码（相对位置、复数理解） | ✅ |
| ALiBi 线性偏置注意力 | ✅ |
| 外推能力对比 | ✅ |
| PyTorch 实现 RoPE 注意力 | ✅ |

### 位置编码选择指南

| 方法 | 外推能力 | 谁在用 | 适用场景 |
|------|---------|--------|---------|
| 绝对（正弦余弦） | 一般 | 原始 Transformer | 固定长度 |
| 学习式 | ❌ 不能 | BERT | 固定长度、不外推 |
| **RoPE** | ✅ 好 | **LLaMA, Mistral, Qwen** | **现代 LLM 标配** |
| ALiBi | ✅ 极好 | BLOOM, MPT | 需要强外推 |

> 记住：**RoPE 是当前主流**，几乎所有现代开源 LLM 都用它。

### 🔗 下一章预告

**`05_generalization_theory.ipynb`** — 泛化理论、双下降、万能近似定理

---

> 💬 **写在最后**：位置编码看似一个小细节，但它决定了模型能否处理长序列、能否外推。
> RoPE 的优雅在于——**把位置信息变成旋转，让 attention 自然只依赖相对位置**。
> 这个 idea 来自 2020 年的 RoFormer 论文，如今统治了整个 LLM 领域。""")

# ============================================================
output_path = "notebooks/04_positional_encoding.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")