"""生成 07_rnn_vs_transformer.ipynb 的脚本"""
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
md("""# 07 — RNN/LSTM ↔ Transformer：序列建模范式转移

> 从 1986 年 RNN 到 1997 年 LSTM，循环网络统治序列建模 30 年。
> 2017 年 Transformer 一篇"Attention Is All You Need"把循环连接**全删**，
> 换成纯自注意力——不仅能并行训练，长距离依赖还更好。
>
> 本章我们实现 RNN/LSTM/Transformer，在长距离依赖任务上对比，亲眼看到这个范式转移。

## 本章你将掌握

1. **RNN 的时序归纳偏置**：隐状态递推 $h_t = f(h_{t-1}, x_t)$
2. **梯度消失/爆炸**：BPTT 为什么让 RNN 记不住长距离
3. **LSTM 门控机制**：遗忘门/输入门/输出门如何缓解梯度消失
4. **Transformer 做序列建模**：自注意力并行 + 全局视野
5. **对比实验**：长距离依赖、并行训练速度、不同序列长度""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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
md("""## 1. RNN：循环神经网络

### 1.1 核心思想：隐状态递推

RNN 处理序列的方式是**维护一个隐状态**，每一步根据当前输入和上一步隐状态更新：

$$h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$$

```
序列: x1 → x2 → x3 → ... → xT
       ↓    ↓    ↓           ↓
隐状态: h1 → h2 → h3 → ... → hT
       (h_t 依赖 h_{t-1}，必须逐步计算)
```

### 1.2 RNN 的时序归纳偏置

| 归纳偏置 | 含义 |
|---------|------|
| **时序性** | 必须按顺序处理，$h_t$ 只看过去 |
| **马尔可夫压缩** | 所有历史压缩进固定大小的 $h_t$ |
| **参数共享** | 每步用同一组权重 |

> 💡 第一个偏置是 RNN 的**优势也是枷锁**——必须逐步计算 → 无法并行。
> 第二个偏置是**信息瓶颈**——长序列的历史被压进固定向量，容易遗忘。""")

code("""# 从零实现一个 RNN cell（教学版）
class SimpleRNNCell:
    \"\"\"RNN cell: h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b)\"\"\"
    def __init__(self, input_size, hidden_size):
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def __call__(self, x, h_prev):
        h = np.tanh(x @ self.W_xh + h_prev @ self.W_hh + self.b)
        return h

    def zero_state(self, batch_size):
        return np.zeros((batch_size, self.hidden_size))

# 处理一个序列
rnn_cell = SimpleRNNCell(input_size=4, hidden_size=8)
sequence = np.random.randn(5, 4)  # 长度5，特征4
h = rnn_cell.zero_state(1)
hidden_states = [h.copy()]
for t in range(5):
    h = rnn_cell(sequence[t:t+1], h)
    hidden_states.append(h.copy())

hidden_states = np.array(hidden_states).squeeze()
print(f"序列长度: 5, 隐状态变化: {hidden_states.shape}")
print("RNN 逐步处理序列——每步依赖上一步，无法并行。")""")

# ============================================================
md("""## 2. RNN 的致命问题：梯度消失/爆炸

### 2.1 BPTT（沿时间反向传播）

RNN 反向传播要沿时间展开，梯度连乘：

$$\\frac{\\partial L}{\\partial h_0} = \\frac{\\partial L}{\\partial h_T} \\prod_{t=1}^{T} \\frac{\\partial h_t}{\\partial h_{t-1}}$$

其中 $\\frac{\\partial h_t}{\\partial h_{t-1}} = \\text{diag}(1-h_t^2) W_{hh}$。

### 2.2 连乘的灾难

- 若 $W_{hh}$ 的特征值 $> 1$ → **梯度爆炸**（NaN）
- 若 $W_{hh}$ 的特征值 $< 1$ → **梯度消失**（早期信息传不到）

```
T=5:  梯度 × 0.9^5  = 0.59  ✅ 还行
T=20: 梯度 × 0.9^20 = 0.12  ⚠️ 衰减
T=50: 梯度 × 0.9^50 = 0.005 ❌ 几乎没了
```

> 这就是 RNN 记不住长距离依赖的数学原因——梯度连乘让早期信息指数衰减。""")

code("""# 可视化梯度消失/爆炸
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 梯度随时间步的衰减
T = 50
timesteps = np.arange(1, T + 1)
for scale, label, color in [(0.5, '特征值=0.5 (消失)', 'blue'),
                              (0.9, '特征值=0.9 (消失)', 'cyan'),
                              (1.0, '特征值=1.0 (稳定)', 'green'),
                              (1.05, '特征值=1.05 (爆炸)', 'red')]:
    grad = scale ** timesteps
    axes[0].plot(timesteps, np.clip(grad, 1e-10, 1e10), label=label, color=color, linewidth=2)

axes[0].set_yscale('log')
axes[0].set_xlabel('时间步 T')
axes[0].set_ylabel('梯度大小 (对数尺度)')
axes[0].set_title('RNN 梯度随时间步的变化')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# PyTorch 验证：测量不同时间步的梯度
hidden_size = 16
results = []
for T in [5, 10, 20, 40]:
    rnn = nn.RNN(input_size=4, hidden_size=hidden_size, batch_first=True)
    x = torch.randn(1, T, 4)
    h0 = torch.zeros(1, 1, hidden_size)
    out, _ = rnn(x, h0)
    loss = out[0, -1, 0]  # 最后一步的第一个输出
    rnn.zero_grad()
    loss.backward()
    # 测量第一层权重的梯度范数
    grad_norm = rnn.weight_hh_l0.grad.norm().item()
    results.append((T, grad_norm))

ts, gs = zip(*results)
axes[1].bar(range(len(ts)), gs, tick_label=[f'T={t}' for t in ts], color='steelblue')
axes[1].set_ylabel('权重梯度范数')
axes[1].set_title('PyTorch RNN: 不同序列长度的梯度')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/fig_rnn_gradient.png', bbox_inches='tight')
plt.show()
print("梯度随序列长度增加而衰减——RNN 记不住长距离依赖。")""")

# ============================================================
md("""## 3. LSTM：门控缓解梯度消失

### 3.1 LSTM 的核心思想

LSTM（1997）的解法是加一个**细胞状态** $c_t$ 和三个**门**：

$$\\begin{aligned}
f_t &= \\sigma(W_f [h_{t-1}, x_t]) & \\text{遗忘门: 决定丢弃多少旧信息} \\\\
i_t &= \\sigma(W_i [h_{t-1}, x_t]) & \\text{输入门: 决定写入多少新信息} \\\\
\\tilde{c}_t &= \\tanh(W_c [h_{t-1}, x_t]) & \\text{候选新信息} \\\\
c_t &= f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t & \\text{细胞更新: 加法不是乘法!} \\\\
o_t &= \\sigma(W_o [h_{t-1}, x_t]) & \\text{输出门} \\\\
h_t &= o_t \\odot \\tanh(c_t) & \\text{隐状态}
\\end{aligned}$$

### 3.2 为什么 LSTM 能记住长距离？

关键在 $c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t$ ——这是**加法**不是乘法！

- RNN: $h_t = \\tanh(W h_{t-1} + ...)$ → 梯度 $\\prod W$ → 连乘消失
- LSTM: $c_t = f_t c_{t-1} + ...$ → 梯度 $\\prod f_t$ → 遗忘门可以学成 $\\approx 1$，让梯度**线性流动**

```
RNN:  h0 → ×W → ×W → ×W → ...  (连乘，消失)
LSTM: c0 → +Δ → +Δ → +Δ → ...  (加法，信息持续累积)
```

> 💡 遗忘门 $f_t \\approx 1$ 时，细胞状态像一条**传送带**，信息可以无损流过很长距离。""")

code("""# 从零实现 LSTM cell（教学版）
class SimpleLSTMCell:
    \"\"\"LSTM cell with 3 gates + cell state\"\"\"
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        # 合并所有门的权重 (4个: f, i, c_candidate, o)
        self.W_x = np.random.randn(input_size, 4 * hidden_size) * 0.1
        self.W_h = np.random.randn(hidden_size, 4 * hidden_size) * 0.1
        self.b = np.zeros(4 * hidden_size)

    def __call__(self, x, h_prev, c_prev):
        gates = x @ self.W_x + h_prev @ self.W_h + self.b
        f = 1 / (1 + np.exp(-gates[:, :self.hidden_size]))          # 遗忘门
        i = 1 / (1 + np.exp(-gates[:, self.hidden_size:2*self.hidden_size]))   # 输入门
        c_bar = np.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])  # 候选
        o = 1 / (1 + np.exp(-gates[:, 3*self.hidden_size:]))        # 输出门

        c = f * c_prev + i * c_bar  # 招加法更新——关键!
        h = o * np.tanh(c)
        return h, c

    def zero_state(self, batch_size):
        return np.zeros((batch_size, self.hidden_size)), np.zeros((batch_size, self.hidden_size))

# 对比 RNN vs LSTM 在长序列上的信息保持
seq_len = 50
input_size = 4
hidden_size = 8

# 在序列开头放一个信号，看最后一步还剩多少
rnn_cell = SimpleRNNCell(input_size, hidden_size)
lstm_cell = SimpleLSTMCell(input_size, hidden_size)

# RNN
h_rnn = rnn_cell.zero_state(1)
h_rnn += np.ones((1, hidden_size))  # 初始信号
signal_rnn = [np.linalg.norm(h_rnn)]
for t in range(seq_len):
    h_rnn = rnn_cell(np.zeros((1, input_size)), h_rnn)  # 后续输入全0
    signal_rnn.append(np.linalg.norm(h_rnn))

# LSTM
h_lstm, c_lstm = lstm_cell.zero_state(1)
c_lstm += np.ones((1, hidden_size))  # 初始信号放细胞状态
signal_lstm = [np.linalg.norm(c_lstm)]
for t in range(seq_len):
    h_lstm, c_lstm = lstm_cell(np.zeros((1, input_size)), h_lstm, c_lstm)
    signal_lstm.append(np.linalg.norm(c_lstm))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(signal_rnn, 'b-o', label='RNN 隐状态', linewidth=2, markersize=4)
ax.plot(signal_lstm, 'r-s', label='LSTM 细胞状态', linewidth=2, markersize=4)
ax.set_xlabel('时间步')
ax.set_ylabel('信号范数')
ax.set_title('RNN vs LSTM：初始信号的保持能力')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_lstm_vs_rnn_memory.png', bbox_inches='tight')
plt.show()
print("LSTM 的细胞状态保持信号远好于 RNN——加法更新是关键。")""")

# ============================================================
md("""## 4. Transformer：纯注意力做序列建模

### 4.1 Transformer 的核心改变

Transformer（2017）做了个激进决定——**删掉所有循环连接**，只用自注意力：

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}}\\right) V$$

```
RNN:        x1 → h1 → h2 → h3 → ... → hT   (逐步，串行)
Transformer: x1, x2, x3, ..., xT → Attention → h1, h2, ..., hT  (一次算完，并行!)
```

### 4.2 Transformer 的优势

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| **并行性** | ❌ 必须逐步计算 | ✅ 所有位置同时算 |
| **长距离依赖** | ⚠️ 梯度衰减 | ✅ 任意两距离都是 O(1) |
| **信息瓶颈** | ❌ 固定大小隐状态 | ✅ 每层看到全部位置 |
| **时序归纳偏置** | ✅ 内置顺序处理 | ❌ 需要位置编码补充 |

> 💡 Transformer 牺牲了时序归纳偏置（需要位置编码补回来），换来了**并行性**和**全局视野**。
> 这个交易在 GPU 并行计算时代极其划算——训练速度提升数倍。""")

code("""# 从零实现多头自注意力（教学版）
class SimpleMultiHeadAttention:
    \"\"\"Multi-head attention: Q, K, V → softmax(QK^T/√d)V\"\"\"
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def __call__(self, x):
        \"\"\"x: (batch, seq_len, d_model) → output: (batch, seq_len, d_model)\"\"\"
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # 重塑成多头: (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        batch, seq_len, _ = x.shape
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # 注意力分数: (batch, n_heads, seq, seq)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        # softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores - scores_max)
        attn = attn / attn.sum(axis=-1, keepdims=True)

        # 应用注意力到 V
        out = attn @ V  # (batch, n_heads, seq, d_k)
        # 合并多头
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        return out @ self.W_o

# 演示: Transformer 一次处理整个序列
attn = SimpleMultiHeadAttention(d_model=16, n_heads=4)
seq = np.random.randn(2, 10, 16)  # batch=2, seq_len=10, d_model=16
output = attn(seq)
print(f"输入: {seq.shape} → 输出: {output.shape}")
print("Transformer 一次处理整个序列——所有位置并行计算！")""")

# ============================================================
md("""## 5. 对比实验：长距离依赖任务

用经典的**复制任务**：给一个序列，让模型记住开头几个 token，在末尾复制出来。
这直接测试模型的长距离记忆能力。""")

code("""def make_copy_task(batch_size, seq_len, n_symbols=8, n_copy=3):
    \"\"\"复制任务: 序列开头 n_copy 个符号要在末尾复制出来
    输入: [c1, c2, c3, 0, 0, ..., 0, -1]  (-1 是提示符)
    目标: [_, _, _, _, _, ..., c1, c2, c3]
    \"\"\"
    x = np.zeros((batch_size, seq_len), dtype=np.int64)
    y = np.zeros((batch_size, seq_len), dtype=np.int64)
    for b in range(batch_size):
        symbols = np.random.randint(1, n_symbols, size=n_copy)
        x[b, :n_copy] = symbols
        x[b, -1] = n_symbols  # 提示符
        y[b, -n_copy:] = symbols
    return x, y

# 可视化任务
x_demo, y_demo = make_copy_task(1, 12, n_copy=3)
print("复制任务示例:")
print(f"  输入: {x_demo[0]}")
print(f"  目标: {y_demo[0]}")
print("  开头的 [c1,c2,c3] 要在末尾复制出来——测试长距离记忆")""")

code("""# RNN/LSTM 模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        return self.fc(out)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out)

# Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*2,
            batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embed(x) + self.pos_embed[:, :seq_len]
        x = self.encoder(x)
        return self.fc(x)

vocab_size = 10  # 0-7 符号 + 0 padding + 9 提示符
print("三个模型定义完成: RNN, LSTM, Transformer")""")

code("""def train_model(model, X, Y, n_copy=3, epochs=200, lr=1e-2):
    \"\"\"训练模型, 只在最后 n_copy 位置计算损失\"\"\"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        # 只在最后 n_copy 位置算损失——逼模型学会复制
        loss = loss_fn(logits[:, -n_copy:].reshape(-1, logits.size(-1)),
                       Y[:, -n_copy:].reshape(-1))
        loss.backward()
        optimizer.step()

def eval_accuracy(model, X, Y, n_copy=3):
    \"\"\"只看最后 n_copy 位置的准确率\"\"\"
    with torch.no_grad():
        logits = model(X)
        pred = logits[:, -n_copy:].argmax(-1)
        target = Y[:, -n_copy:]
        return (pred == target).all(dim=1).float().mean().item()

# 不同序列长度对比
seq_lengths = [10, 20, 30, 50]
results = {'RNN': [], 'LSTM': [], 'Transformer': []}
n_copy = 3

for seq_len in seq_lengths:
    X_np, Y_np = make_copy_task(128, seq_len, n_copy=n_copy)
    X_t = torch.tensor(X_np)
    Y_t = torch.tensor(Y_np)

    for name, ModelClass in [('RNN', RNNModel), ('LSTM', LSTMModel), ('Transformer', TransformerModel)]:
        torch.manual_seed(42)
        model = ModelClass(vocab_size)
        train_model(model, X_t, Y_t, n_copy=n_copy, epochs=300, lr=1e-2)
        acc = eval_accuracy(model, X_t, Y_t, n_copy)
        results[name].append(acc)

    print(f"序列长度={seq_len:3d}: RNN={results['RNN'][-1]:.1%}, "
          f"LSTM={results['LSTM'][-1]:.1%}, Transformer={results['Transformer'][-1]:.1%}")""")

code("""# 可视化对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(seq_lengths, results['RNN'], 'b-o', linewidth=2.5, markersize=8, label='RNN')
ax.plot(seq_lengths, results['LSTM'], 'g-s', linewidth=2.5, markersize=8, label='LSTM')
ax.plot(seq_lengths, results['Transformer'], 'r-^', linewidth=2.5, markersize=8, label='Transformer')
ax.set_xlabel('序列长度')
ax.set_ylabel('复制准确率 (最后3位全对)')
ax.set_title('长距离依赖: RNN vs LSTM vs Transformer')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('notebooks/fig_seq_comparison.png', bbox_inches='tight')
plt.show()
print("序列越长, RNN 越记不住; LSTM 靠门控好一些; Transformer 全局注意力最稳。")""")

# ============================================================
md("""## 6. 并行性对比：训练速度

RNN 必须逐步计算，Transformer 可以一次处理整个序列。在 GPU 上这个差异巨大。""")

code("""# 测量不同序列长度下的前向传播时间
seq_lengths = [32, 64, 128, 256, 512]
timing = {'RNN': [], 'LSTM': [], 'Transformer': []}

for seq_len in seq_lengths:
    X = torch.randint(0, vocab_size, (32, seq_len))

    for name, ModelClass in [('RNN', RNNModel), ('LSTM', LSTMModel), ('Transformer', TransformerModel)]:
        model = ModelClass(vocab_size)
        # warmup
        for _ in range(3):
            _ = model(X)
        # measure
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = model(X)
        elapsed = (time.time() - start) / 10 * 1000  # ms
        timing[name].append(elapsed)

    print(f"序列长度={seq_len:4d}: RNN={timing['RNN'][-1]:6.1f}ms, "
          f"LSTM={timing['LSTM'][-1]:6.1f}ms, "
          f"Transformer={timing['Transformer'][-1]:6.1f}ms")""")

code("""# 可视化训练速度对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(seq_lengths, timing['RNN'], 'b-o', linewidth=2.5, markersize=8, label='RNN')
ax.plot(seq_lengths, timing['LSTM'], 'g-s', linewidth=2.5, markersize=8, label='LSTM')
ax.plot(seq_lengths, timing['Transformer'], 'r-^', linewidth=2.5, markersize=8, label='Transformer')
ax.set_xlabel('序列长度')
ax.set_ylabel('前向传播时间 (ms)')
ax.set_title('并行性对比: 前向传播速度')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_parallel_speed.png', bbox_inches='tight')
plt.show()
print("RNN/LSTM 随序列长度线性增长; Transformer 并行度高, 增长更缓。")""")

# ============================================================
md("""## 7. 范式转移的核心张力

### 7.1 总结

```
            时序归纳偏置强                    时序归纳偏置弱
            ←─────────────────────────────────→
   RNN          LSTM                          Transformer
 (逐步递推)    (门控缓解)                     (纯注意力)

   串行训练 ❌   串行训练 ❌                    并行训练 ✅
   短距离 ✅     中距离 ✅                      长距离 ✅
   长距离 ❌     长距离 ⚠️                     全局视野 ✅
```

### 7.2 为什么 Transformer 赢了？

| 原因 | 说明 |
|------|------|
| **GPU 并行时代** | RNN 的时序依赖无法利用 GPU 并行；Transformer 完美并行 |
| **长距离依赖** | 自注意力任意两位置距离都是 O(1)，RNN 是 O(T) |
| **信息无瓶颈** | 每层每个位置都能看到所有其他位置，不压缩进固定向量 |
| **scaling 友好** | 并行训练 → 更大数据 → 更好性能 → 更多算力投入 |

### 7.3 RNN 没有完全死

Transformer 也有弱点：序列长度 $T$ 的注意力是 $O(T^2)$ 复杂度。于是：

- **线性注意力**：把 softmax 近似成核函数，降到 $O(T)$
- **Mamba/SSM**：状态空间模型，有 RNN 的线性复杂度 + Transformer 的并行训练
- **RWKV**：RNN 式推理但 Transformer 式训练

> 这些将在下一章 `08_architecture_innovations.ipynb` 详讲——它们试图**结合 RNN 的效率和 Transformer 的质量**。""")

code("""# 可视化复杂度对比
fig, ax = plt.subplots(figsize=(10, 6))
T = np.arange(1, 200)
ax.plot(T, T, 'g-', linewidth=3, label='RNN/LSTM/线性注意力: O(T)')
ax.plot(T, T * T, 'r-', linewidth=3, label='Transformer 自注意力: O(T²)')
ax.fill_between(T, T, T*T, where=T*T>T, color='red', alpha=0.1)
ax.set_xlabel('序列长度 T')
ax.set_ylabel('计算复杂度')
ax.set_title('复杂度对比: 线性 vs 二次')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 200); ax.set_ylim(0, 20000)
plt.tight_layout()
plt.savefig('notebooks/fig_complexity.png', bbox_inches='tight')
plt.show()
print("Transformer 的 O(T²) 是它的阿喀琉斯之踵——长序列时计算量爆炸。")
print("这正是 Mamba/线性注意力 等新架构要解决的问题。")""")

# ============================================================
md("""## 8. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| RNN 隐状态递推与时序归纳偏置 | ✅ |
| BPTT 梯度消失/爆炸的数学原因 | ✅ |
| LSTM 门控机制（遗忘门/输入门/输出门） | ✅ |
| LSTM 加法更新为什么能记住长距离 | ✅ |
| Transformer 自注意力做序列建模 | ✅ |
| 并行性 vs 时序归纳偏置的权衡 | ✅ |
| 复杂度 O(T) vs O(T²) | ✅ |

### 核心 takeaway

> **RNN 的时序归纳偏置在 GPU 并行时代成了枷锁**。
> Transformer 牺牲时序先验换来并行性 + 全局视野，这个交易在算力时代极其划算。
> 但 O(T²) 复杂度催生了 Mamba/线性注意力等新架构——**序列建模的故事还没结束**。

### 🔗 下一章预告

**`08_architecture_innovations.ipynb`** — MoE/SSM-Mamba/线性注意力/液态网络 + GQA/MQA

---

> 💬 **写在最后**：RNN→Transformer 的转移不只是架构变化，更是**从"时序处理"到"全局并行"**的范式革命。
> 理解这个转移，就理解了为什么 LLM 都用 Transformer，以及为什么 Mamba 试图挑战它。""")

# ============================================================
output_path = "notebooks/07_rnn_vs_transformer.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")