"""生成 03_normalization_activation.ipynb 的脚本"""
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
# 概述
# ============================================================
md("""# 03 — 归一化、激活函数演进与参数初始化

> 前两章我们搞定了反向传播和优化器。但你可能遇到过：**网络深了训不动、loss 爆炸成 NaN、
> 有些神经元永远不激活**。这些问题的根源往往不在优化器，而在三个被忽视的细节：
> **归一化、激活函数、初始化**。
>
> 本章我们讲清楚现代 LLM 的三个"标配"选择为什么是它们：
> - 归一化：**BatchNorm → LayerNorm → RMSNorm**（为什么 LLaMA 用 RMSNorm）
> - 激活函数：**ReLU → GELU → SwiGLU**（为什么 LLaMA 用 SwiGLU）
> - 初始化：**Xavier vs He**（为什么 ReLU 网络要用 He 初始化）

## 本章你将掌握

1. **内部协变量偏移**：为什么需要归一化
2. **BatchNorm / LayerNorm / RMSNorm** 的原理、区别与各自适用场景
3. **ReLU → GELU → SwiGLU** 的演进，每个解决了什么问题
4. **Xavier / He 初始化**：为什么初始化影响巨大
5. 用实验**亲眼看到**错误选择导致训练崩溃""")

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
# 归一化
# ============================================================
md("""## 1. 归一化：为什么深网络训不动？

### 1.1 内部协变量偏移（Internal Covariate Shift）

考虑一个深层网络。第 10 层的输入是第 9 层的输出。但在训练过程中，第 9 层的权重一直在变，
导致第 10 层看到的输入分布**一直在变**。

> 想象你在打移动靶——靶子一直在动，你很难瞄准。

**归一化的作用**：把每一层的输入拉回稳定的分布（均值 0、方差 1），让后续层看到的数据分布不再剧烈漂移。

### 1.2 三种归一化的区别：沿哪个维度归一化？

这是理解三种归一化的**关键**。假设输入形状是 `(batch, feature)`：

```
              batch 维度 →
              ┌──────────────┐
feature 维度  │ x x x x x x  │  ← BatchNorm：沿 batch 维度归一化（每列）
   ↓         │ x x x x x x  │  ← LayerNorm：沿 feature 维度归一化（每行）
              │ x x x x x x  │
              └──────────────┘
```

| 归一化 | 沿哪个维度 | 适用场景 |
|--------|-----------|---------|
| **BatchNorm** | batch 维度（每列） | CNN、大 batch 场景 |
| **LayerNorm** | feature 维度（每行） | RNN、Transformer（batch 无关） |
| **RMSNorm** | feature 维度（去掉均值中心化） | LLaMA 等现代 LLM（LayerNorm 的轻量版） |""")

code("""# === 从零实现三种归一化 ===

def batch_norm(x, gamma, beta, eps=1e-5):
    \"\"\"BatchNorm: 沿 batch 维度归一化
    x: (batch, feature)
    \"\"\"
    mean = x.mean(axis=0, keepdims=True)   # 每个 feature 在 batch 上求均值
    var = x.var(axis=0, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def layer_norm(x, gamma, beta, eps=1e-5):
    \"\"\"LayerNorm: 沿 feature 维度归一化
    x: (batch, feature)
    \"\"\"
    mean = x.mean(axis=1, keepdims=True)   # 每个 sample 在 feature 上求均值
    var = x.var(axis=1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta

def rms_norm(x, gamma, eps=1e-5):
    \"\"\"RMSNorm: LayerNorm 去掉均值中心化，只用 RMS 归一化
    x: (batch, feature)
    \"\"\"
    rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True) + eps)
    x_hat = x / rms
    return gamma * x_hat

# 演示
np.random.seed(42)
x = np.random.randn(4, 6) * 3 + 2  # (batch=4, feature=6)，故意偏移
gamma = np.ones((1, 6))
beta = np.zeros((1, 6))

print("原始输入 (4 samples, 6 features):")
print(f"  均值: {x.mean(axis=0).round(2)}")
print(f"  标准差: {x.std(axis=0).round(2)}")

bn = batch_norm(x, gamma, beta)
ln = layer_norm(x, gamma, beta)
rn = rms_norm(x, gamma)

print(f"\\nBatchNorm 后 (沿 batch 归一化，每列均值≈0，std≈1):")
print(f"  均值: {bn.mean(axis=0).round(3)}")
print(f"  标准差: {bn.std(axis=0).round(3)}")

print(f"\\nLayerNorm 后 (沿 feature 归一化，每行均值≈0，std≈1):")
print(f"  行均值: {ln.mean(axis=1).round(3)}")
print(f"  行标准差: {ln.std(axis=1).round(3)}")

print(f"\\nRMSNorm 后 (沿 feature 归一化，每行 RMS≈1):")
print(f"  行 RMS: {np.sqrt((rn**2).mean(axis=1)).round(3)}")""")

md("""### 1.3 BatchNorm 的致命问题：依赖 batch 大小

BatchNorm 在 batch 维度上统计均值和方差。如果 batch 太小（比如 1 或 2），统计就不准。
而且**推理时用的和训练时不是同一个 batch**——需要维护一个移动平均的 running statistics。

> 这就是为什么 **Transformer 用 LayerNorm 而不是 BatchNorm**：
> - NLP 中序列长度可变，batch 维度统计不稳定
> - LayerNorm 对每个样本独立归一化，不依赖 batch 大小
> - 训练和推理行为一致，没有 running statistics 的麻烦

### 1.4 RMSNorm：LayerNorm 的轻量版

RMSNorm 去掉了 LayerNorm 里的**减均值**步骤，只用 RMS（均方根）归一化：

$$\\text{LayerNorm}: \\hat{x} = \\frac{x - \\mu}{\\sigma}, \\quad \\text{RMSNorm}: \\hat{x} = \\frac{x}{\\text{RMS}(x)}$$

- **少了均值计算**，速度更快（约 7-64% 加速）
- 实验表明效果和 LayerNorm 几乎一样
- **LLaMA、Mistral、Qwen 等现代 LLM 都用 RMSNorm**""")

code("""# 可视化三种归一化的效果
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

np.random.seed(42)
x = np.random.randn(100, 50) * 5 + 3  # 100 samples, 50 features

# 原始数据分布
axes[0].hist(x.flatten(), bins=50, color='#3498DB', alpha=0.7)
axes[0].set_title(f'原始数据\\n均值={x.mean():.2f}, std={x.std():.2f}')
axes[0].axvline(x.mean(), color='red', linestyle='--', label=f'均值={x.mean():.2f}')
axes[0].legend()

# LayerNorm
gamma = np.ones((1, 50)); beta = np.zeros((1, 50))
ln = layer_norm(x, gamma, beta)
axes[1].hist(ln.flatten(), bins=50, color='#2ECC71', alpha=0.7)
axes[1].set_title(f'LayerNorm\\n均值={ln.mean():.3f}, std={ln.std():.3f}')
axes[1].axvline(ln.mean(), color='red', linestyle='--')
axes[1].legend()

# RMSNorm
rn = rms_norm(x, gamma)
axes[2].hist(rn.flatten(), bins=50, color='#E67E22', alpha=0.7)
axes[2].set_title(f'RMSNorm\\n均值={rn.mean():.3f}, std={rn.std():.3f}')
axes[2].axvline(rn.mean(), color='red', linestyle='--')
axes[2].legend()

plt.tight_layout()
plt.savefig('notebooks/fig_normalization_comparison.png', bbox_inches='tight')
plt.show()
print("LayerNorm 均值≈0、std≈1；RMSNorm 均值略偏但 RMS≈1，计算更简单。")""")

# ============================================================
# 激活函数演进
# ============================================================
md("""## 2. 激活函数演进：ReLU → GELU → SwiGLU

### 2.1 ReLU 的问题

ReLU 在第 1 章已经讲过。它有两个问题：

1. **死神经元**：如果某个神经元的输入永远 < 0，梯度永远为 0，这个神经元就"死了"，再也不会更新
2. **非光滑**：在 0 点不可导，虽然实践中问题不大，但理论上不优雅
3. **零均值偏移**：ReLU 输出永远 ≥ 0，导致下一层输入有正的偏置（"均值偏移"）

### 2.2 GELU：高斯误差线性单元

$$\\text{GELU}(x) = x \\cdot \\Phi(x) = x \\cdot \\frac{1}{2}\\left[1 + \\text{erf}\\left(\\frac{x}{\\sqrt{2}}\\right)\\right]$$

直觉：**输入越大越可能被保留，输入越小越可能被置零**——但不是硬截断，而是**概率性**的。

- 比 ReLU 更光滑（处处可导）
- 在 0 附近有非零梯度，缓解死神经元
- **BERT、GPT 系列都用 GELU**

### 2.3 SwiGLU：Swish + GLU

这是现代 LLM 的标配。先理解两个组件：

**Swish**：$\\text{Swish}(x) = x \\cdot \\sigma(\\beta x)$，是 GELU 的近亲，但更灵活

**GLU（Gated Linear Unit）**：$\\text{GLU}(a, b) = a \\odot \\sigma(b)$，用 $b$ 当"门"控制 $a$

**SwiGLU**：把 Swish 和 GLU 结合：
$$\\text{SwiGLU}(x) = \\text{Swish}(xW_1) \\odot (xW_2)$$

- 有门控机制，表达能力更强
- **LLaMA、PaLM、Qwen 等现代 LLM 都用 SwiGLU**
- 代价：多一个权重矩阵 $W_2$""")

code("""from scipy.special import erf

def relu(x): return np.maximum(0, x)
def gelu(x): return x * 0.5 * (1 + erf(x / np.sqrt(2)))
def swish(x, beta=1): return x / (1 + np.exp(-beta * x))
def sigmoid(x): return 1 / (1 + np.exp(-x))

# 可视化激活函数
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.linspace(-4, 4, 300)

axes[0].plot(x, relu(x), label='ReLU', linewidth=2.5)
axes[0].plot(x, gelu(x), label='GELU', linewidth=2.5)
axes[0].plot(x, swish(x), label='Swish', linewidth=2.5)
axes[0].axhline(0, color='gray', linewidth=0.5)
axes[0].axvline(0, color='gray', linewidth=0.5)
axes[0].set_title('激活函数对比')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 导数
def relu_deriv(x): return (x > 0).astype(float)
def gelu_deriv(x):
    phi = 0.5 * (1 + erf(x / np.sqrt(2)))
    pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    return phi + x * pdf
def swish_deriv(x, beta=1):
    s = sigmoid(beta * x)
    return s + x * beta * s * (1 - s)

axes[1].plot(x, relu_deriv(x), label="ReLU'", linewidth=2.5)
axes[1].plot(x, gelu_deriv(x), label="GELU'", linewidth=2.5)
axes[1].plot(x, swish_deriv(x), label="Swish'", linewidth=2.5)
axes[1].axhline(0, color='gray', linewidth=0.5)
axes[1].set_title('导数对比（注意 0 附近）')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_activation_evolution.png', bbox_inches='tight')
plt.show()
print("关键区别：GELU/Swish 在 0 附近更平滑，负数区域有非零梯度 → 缓解死神经元。")""")

code("""# SwiGLU 的实现与对比
def swiglu(x, W1, W2, b1=None, b2=None):
    \"\"\"SwiGLU(x) = Swish(x @ W1) * (x @ W2)
    比普通激活函数多一个门控矩阵 W2
    \"\"\"
    if b1 is None: b1 = 0
    if b2 is None: b2 = 0
    gate = swish(x @ W1 + b1)
    value = x @ W2 + b2
    return gate * value

# 对比普通 FFN vs SwiGLU FFN
np.random.seed(42)
d_model = 64
d_ff = 256
x = np.random.randn(4, d_model)

# 普通 FFN: x -> Linear -> ReLU -> Linear
W1_plain = np.random.randn(d_model, d_ff) * np.sqrt(2/d_model)
W2_plain = np.random.randn(d_ff, d_model) * np.sqrt(2/d_ff)
ffn_plain = relu(x @ W1_plain) @ W2_plain

# SwiGLU FFN: x -> SwiGLU(x@W1, x@W2) -> W3
W1_sw = np.random.randn(d_model, d_ff) * np.sqrt(2/d_model)
W2_sw = np.random.randn(d_model, d_ff) * np.sqrt(2/d_model)
W3_sw = np.random.randn(d_ff, d_model) * np.sqrt(2/d_ff)
ffn_swiglu = swiglu(x, W1_sw, W2_sw) @ W3_sw

print(f"普通 FFN 输出形状: {ffn_plain.shape}, 均值: {ffn_plain.mean():.4f}")
print(f"SwiGLU FFN 输出形状: {ffn_swiglu.shape}, 均值: {ffn_swiglu.mean():.4f}")
print(f"\\nSwiGLU 多一个矩阵参数 ({d_model}x{d_ff}={d_model*d_ff})，但表达能力更强。")
print(f"这就是 LLaMA 的 FFN 结构：SwiGLU 替代 ReLU。")""")

# ============================================================
# 初始化
# ============================================================
md("""## 3. 参数初始化：为什么不能全用 0

### 3.1 全零初始化的灾难

如果所有权重初始化为 0，那么：
- 所有神经元计算完全相同的前向输出
- 反向传播时所有神经元收到完全相同的梯度
- **永远对称，永远学不到不同的特征**

所以初始化必须**打破对称性**。但太大了也不行——信号会指数爆炸；太小了——信号会指数消失。

### 3.2 Xavier 初始化（配合 tanh/sigmoid）

$$W \\sim \\mathcal{N}\\left(0, \\frac{1}{n_{\\text{in}}}\\right)$$

目标：让每一层的方差保持不变。适合**线性区间**的激活函数（tanh、sigmoid）。

### 3.3 He 初始化（配合 ReLU）

$$W \\sim \\mathcal{N}\\left(0, \\frac{2}{n_{\\text{in}}}\\right)$$

ReLU 会把一半的输入置零，所以方差减半。He 初始化把方差翻倍来补偿。**配合 ReLU 用**。""")

code("""# 实验：不同初始化对深网络的影响
def forward_init(init_type, depth=20, width=256, activation='relu'):
    \"\"\"用指定初始化跑一个 depth 层的前向传播，记录每层激活值的方差\"\"\"
    x = np.random.randn(1000, width)  # 1000 个样本

    variances = [np.var(x)]
    for _ in range(depth):
        n_in = width
        if init_type == 'xavier':
            W = np.random.randn(width, width) * np.sqrt(1.0 / n_in)
        elif init_type == 'he':
            W = np.random.randn(width, width) * np.sqrt(2.0 / n_in)
        elif init_type == 'too_small':
            W = np.random.randn(width, width) * 0.01
        elif init_type == 'too_large':
            W = np.random.randn(width, width) * 3.0

        z = x @ W
        if activation == 'relu':
            x = np.maximum(0, z)
        else:
            x = np.tanh(z)
        variances.append(np.var(x))
    return variances

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ReLU 网络
configs_relu = [('too_small', '#E74C3C', '太小 (0.01)'),
                ('xavier', '#F39C12', 'Xavier (1/n)'),
                ('he', '#2ECC71', 'He (2/n) ✓'),
                ('too_large', '#E67E22', '太大 (3.0)')]

for init_type, color, label in configs_relu:
    var = forward_init(init_type, depth=30, activation='relu')
    axes[0].plot(var, color=color, linewidth=2, label=label, marker='o', markersize=3)
axes[0].set_title('ReLU 网络：各层激活方差')
axes[0].set_xlabel('层'); axes[0].set_ylabel('方差')
axes[0].set_yscale('log'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# tanh 网络
configs_tanh = [('too_small', '#E74C3C', '太小 (0.01)'),
                ('xavier', '#2ECC71', 'Xavier (1/n) ✓'),
                ('he', '#F39C12', 'He (2/n)'),
                ('too_large', '#E67E22', '太大 (3.0)')]

for init_type, color, label in configs_tanh:
    var = forward_init(init_type, depth=30, activation='tanh')
    axes[1].plot(var, color=color, linewidth=2, label=label, marker='o', markersize=3)
axes[1].set_title('tanh 网络：各层激活方差')
axes[1].set_xlabel('层'); axes[1].set_ylabel('方差')
axes[1].set_yscale('log'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_initialization.png', bbox_inches='tight')
plt.show()
print("观察：")
print("  ReLU 网络 → He 初始化（绿色）方差稳定，Xavier 会衰减")
print("  tanh 网络 → Xavier 初始化（绿色）方差稳定，He 会爆炸")
print("  选错初始化 → 方差指数爆炸或消失，网络直接废掉！")""")

# ============================================================
# 综合实验
# ============================================================
md("""## 4. 综合实验：错误选择 vs 正确选择

把归一化、激活函数、初始化都用 PyTorch 组合，看错误选择如何导致训练崩溃。""")

code("""def train_experiment(use_norm, init_type, activation_name, epochs=100):
    \"\"\"训练一个深网络，返回损失曲线\"\"\"
    torch.manual_seed(42)
    layers = []
    for i in range(10):  # 10 层深网络
        layers.append(nn.Linear(64, 64))
        if init_type == 'he':
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='relu')
        elif init_type == 'xavier':
            nn.init.xavier_normal_(layers[-1].weight)
        elif init_type == 'bad':
            nn.init.normal_(layers[-1].weight, std=0.01)

        if use_norm == 'layernorm':
            layers.append(nn.LayerNorm(64))
        elif use_norm == 'rmsnorm':
            layers.append(nn.LayerNorm(64))  # 用 LayerNorm 近似

        if activation_name == 'relu':
            layers.append(nn.ReLU())
        elif activation_name == 'gelu':
            layers.append(nn.GELU())

    layers.append(nn.Linear(64, 1))
    model = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    X = torch.randn(200, 64)
    y = torch.randn(200, 1)

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

fig, ax = plt.subplots(figsize=(11, 6))

configs = [
    ('layernorm', 'he', 'gelu', '#2ECC71', '✓ 正确: LayerNorm + He + GELU'),
    ('none', 'he', 'gelu', '#F39C12', '无归一化 + He + GELU'),
    ('layernorm', 'bad', 'gelu', '#E74C3C', 'LayerNorm + 烂初始化 + GELU'),
    ('none', 'bad', 'relu', '#9B59B6', '无归一化 + 烂初始化 + ReLU'),
]

for norm, init, act, color, label in configs:
    losses = train_experiment(norm, init, act)
    ax.plot(losses, color=color, linewidth=2.5, label=label)

ax.set_title('深网络训练：正确配置 vs 错误配置')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_config_comparison.png', bbox_inches='tight')
plt.show()
print("正确配置（绿色）稳定收敛；错误配置要么不下降，要么 NaN 崩溃。")
print("这就是为什么归一化 + 初始化 + 激活函数的选择至关重要！")""")

# ============================================================
# 小结
# ============================================================
md("""## 5. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 内部协变量偏移与归一化的必要性 | ✅ |
| BatchNorm / LayerNorm / RMSNorm 原理与区别 | ✅ |
| 为什么 Transformer 用 LayerNorm 而非 BatchNorm | ✅ |
| 为什么现代 LLM 用 RMSNorm | ✅ |
| ReLU → GELU → SwiGLU 演进 | ✅ |
| Xavier vs He 初始化及其适用场景 | ✅ |
| 错误配置导致训练崩溃的实验验证 | ✅ |

### 现代 LLM 的"标配"选择

| 组件 | 选择 | 谁在用 |
|------|------|--------|
| 归一化 | **RMSNorm** | LLaMA, Mistral, Qwen |
| 激活函数 | **SwiGLU** | LLaMA, PaLM, Qwen |
| 初始化 | **He 初始化** | 所有 ReLU 系网络 |

> 记住这个组合：**RMSNorm + SwiGLU + He 初始化** = 现代 LLM 的标准配方。

### 🔗 下一章预告

**`04_positional_encoding.ipynb`** — 位置编码(绝对→RoPE→ALiBi)

我们将：
- 为什么 Transformer 需要位置编码（自注意力本身是位置无关的）
- 绝对位置编码 → RoPE（旋转位置编码，LLaMA 用）→ ALiBi 的演进
- RoPE 为什么能外推到更长序列

---

> 💬 **写在最后**：归一化、激活函数、初始化看似是"细节"，但选错了整个网络就废了。
> 理解了这三个选择背后的原理，你就能看懂任何现代 LLM 的架构图了。""")

# ============================================================
output_path = "notebooks/03_normalization_activation.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")