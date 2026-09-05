"""生成 08_architecture_innovations.ipynb 的脚本"""
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
md("""# 08 — 架构创新：MoE / SSM-Mamba / 线性注意力 / GQA

> Transformer 赢了，但它有 $O(T^2)$ 复杂度和巨大参数量。
> 2022-2026 年的架构创新围绕两个方向：**让注意力更高效** 和 **用非注意力替代**。
>
> 本章覆盖四类创新：
> - **GQA/MQA**：共享 KV 头，减少 KV cache（DeepSpeed/Llama2-3 用的）
> - **MoE**：稀疏激活，参数多但计算少（Mixtral/DeepSeek 用的）
> - **线性注意力**：把 $O(T^2)$ 降到 $O(T)$（Linear Transformer/Performer）
> - **SSM-Mamba**：状态空间模型，RNN 的线性复杂度 + Transformer 的并行训练

## 本章你将掌握

1. **GQA/MQA**：分组/多查询注意力，KV cache 压缩
2. **MoE**：混合专家，top-k 路由 + 负载均衡
3. **线性注意力**：核技巧把 softmax 注意力拆解
4. **SSM/Mamba**：状态空间模型 + 选择性机制""")

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
md("""## 1. GQA / MQA：共享 KV 头

### 1.1 标准 MHA 的 KV cache 问题

标准多头注意力（MHA）有 $n_{heads}$ 个 Q 头、$n_{heads}$ 个 K 头、$n_{heads}$ 个 V 头。
推理时每个头都要缓存自己的 K 和 V → **KV cache 巨大**。

```
MHA:  Q1,K1,V1  Q2,K2,V2  Q3,K3,V3  Q4,K4,V4   (4组KV)
MQA:  Q1, K ,V   Q2, K ,V   Q3, K ,V   Q4, K ,V   (1组KV, 所有Q共享)
GQA:  Q1,K1,V1  Q2,K1,V1  Q3,K2,V2  Q4,K2,V2   (2组KV, 分组共享)
```

### 1.2 MQA 和 GQA

| 方案 | KV 头数 | KV cache | 质量 |
|------|---------|----------|------|
| **MHA** | $n_{heads}$ | 100% | 最好 |
| **GQA** | $n_{groups}$ | $n_{groups}/n_{heads}$ | 接近 MHA |
| **MQA** | 1 | $1/n_{heads}$ | 略降 |

> 💡 GQA 是 MHA 和 MQA 的插值——Llama2-70B 用 8 组 GQA，KV cache 压缩 8x，质量几乎不降。""")

code("""class MultiHeadAttention(nn.Module):
    \"\"\"标准 MHA: 每个Q头有自己的K,V头\"\"\"
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.W_o(out)

class GroupedQueryAttention(nn.Module):
    \"\"\"GQA: n_heads个Q头共享n_kv_heads个KV头\"\"\"
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        Q = self.W_q(x).reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).reshape(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).reshape(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        # 把 KV 头扩展到 Q 头数 (repeat_interleave)
        rep = self.n_heads // self.n_kv_heads
        K = K.repeat_interleave(rep, dim=1)
        V = V.repeat_interleave(rep, dim=1)
        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.W_o(out)

# 对比 KV cache 大小
d_model, n_heads, seq_len, batch = 64, 8, 100, 1
x = torch.randn(batch, seq_len, d_model)

mha = MultiHeadAttention(d_model, n_heads)
mqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads=1)  # MQA: 1 KV head
gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads=4)  # GQA: 4 KV heads

# KV cache 大小 = n_kv_heads * d_k * seq_len * 2 (K and V)
kv_mha = n_heads * (d_model // n_heads) * seq_len * 2
kv_gqa = 4 * (d_model // n_heads) * seq_len * 2
kv_mqa = 1 * (d_model // n_heads) * seq_len * 2

print(f"MHA: {kv_mha} 元素 (100%)")
print(f"GQA: {kv_gqa} 元素 ({kv_gqa/kv_mha:.0%})")
print(f"MQA: {kv_mqa} 元素 ({kv_mqa/kv_mha:.0%})")
print(f"\\n输出验证: MHA={mha(x).shape}, GQA={gqa(x).shape}, MQA={mqa(x).shape}")""")

# ============================================================
md("""## 2. MoE：混合专家

### 2.1 MoE 的核心思想

普通网络每个 token 激活**所有参数**。MoE 让每个 token 只激活 **top-k 个专家**：

$$\\text{MoE}(x) = \\sum_{i \\in \\text{top-k}} \\text{gate}_i(x) \\cdot \\text{Expert}_i(x)$$

```
普通 FFN:  x → [大FFN] → y        (全部参数参与)
MoE:       x → router → 选2个专家  (只有2/8参数参与)
              ↓
           [E1][E2][E3][E4][E5][E6][E7][E8]
            ✓       ✓
```

### 2.2 MoE 的优势

| 特性 | 说明 |
|------|------|
| **参数多，计算少** | 8B 参数只激活 2B → 等效 2B 的计算量 |
| **专家分工** | 不同专家学不同子任务 |
| **挑战：负载均衡** | 如果所有 token 都选同一个专家，其他白费 |

> 💡 Mixtral 8x7B：8 个专家每次选 2 个，总参数 47B 但每次只算 13B。
> DeepSeek-V3 用 256 个专家，更细粒度的路由。""")

code("""class Expert(nn.Module):
    \"\"\"单个专家: 就是一个普通FFN\"\"\"
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))

class MoELayer(nn.Module):
    \"\"\"MoE: router选top-k专家, 加权组合\"\"\"
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(n_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape
        # 1. 路由: 每个token选top-k专家
        gate_logits = self.router(x)  # (B, T, n_experts)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)  # 归一化top-k权重

        # 2. 计算被选中的专家输出
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = indices[..., i]  # (B, T)
            w = weights[..., i].unsqueeze(-1)  # (B, T, 1)
            # 对每个专家, 只处理选到它的token
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    selected = x[mask]
                    expert_out = self.experts[e](selected)
                    output[mask] += w[mask] * expert_out
        return output, gate_logits

def load_balancing_loss(gate_logits, n_experts, top_k):
    \"\"\"鼓励专家被均匀选择的辅助损失\"\"\"
    # 每个专家被选中的概率
    probs = F.softmax(gate_logits, dim=-1).mean(dim=[0, 1])  # (n_experts,)
    # 理想情况: 每个专家被选 top_k/n_experts
    target = top_k / n_experts
    return n_experts * ((probs - target) ** 2).sum()

# 演示 MoE
d_model, d_ff = 64, 128
moe = MoELayer(d_model, d_ff, n_experts=8, top_k=2)
x = torch.randn(2, 10, d_model)
out, logits = moe(x)
lb_loss = load_balancing_loss(logits, 8, 2)

# 统计激活参数
total_params = sum(p.numel() for p in moe.parameters())
# 每次只激活 top_k 个专家
expert_params = sum(p.numel() for p in moe.experts[0].parameters())
active_params = expert_params * 2 + sum(p.numel() for p in moe.router.parameters())

print(f"MoE 总参数: {total_params:,}")
print(f"每次激活参数: {active_params:,} ({active_params/total_params:.1%})")
print(f"负载均衡损失: {lb_loss.item():.4f}")
print(f"输出: {out.shape}")""")

code("""# 可视化专家路由分布
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 训练前: 随机路由
gate_probs_before = F.softmax(logits, dim=-1).mean(dim=[0, 1]).detach().numpy()
axes[0].bar(range(8), gate_probs_before, color='steelblue')
axes[0].set_title('路由分布 (训练前)')
axes[0].set_xlabel('专家编号'); axes[0].set_ylabel('被选概率')
axes[0].set_ylim(0, 0.3); axes[0].grid(True, alpha=0.3, axis='y')

# 模拟训练后: 更均匀
gate_probs_after = np.ones(8) / 8 + np.random.randn(8) * 0.01
gate_probs_after = np.clip(gate_probs_after, 0, None)
gate_probs_after /= gate_probs_after.sum()
axes[1].bar(range(8), gate_probs_after, color='green')
axes[1].axhline(2/8, color='red', linestyle='--', label='理想 (top_k/n_experts)')
axes[1].set_title('路由分布 (训练后+负载均衡)')
axes[1].set_xlabel('专家编号'); axes[1].set_ylabel('被选概率')
axes[1].set_ylim(0, 0.3); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/fig_moe_routing.png', bbox_inches='tight')
plt.show()
print("负载均衡损失让专家被均匀使用——避免某些专家过载、其他闲置。")""")

# ============================================================
md("""## 3. 线性注意力：O(T²) → O(T)

### 3.1 标准注意力的瓶颈

$$\\text{Attn}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}}\\right) V$$

$QK^T$ 是 $(T, T)$ 矩阵 → $O(T^2)$ 复杂度。序列长 100K 时计算量爆炸。

### 3.2 线性注意力的技巧

把 softmax 拆成核函数 $\\phi$：

$$\\text{Attn}(Q,K,V) = \\frac{\\phi(Q) (\\phi(K)^T V)}{\\phi(Q) \\phi(K)^T \\mathbf{1}}$$

关键改变：**先算 $\\phi(K)^T V$** → $(d, d)$ 矩阵，与 $T$ 无关！

```
标准:  Q × K^T → (T,T) × V → O(T²·d)     先算T×T
线性:  K^T × V → (d,d) → Q × (d,d) → O(T·d²)  先算d×d
```

当 $T > d$ 时，线性注意力更快。

> 💡 代价是 $\\phi$ 不能完美近似 softmax，质量略降。
> Performer 用随机特征近似；Linear Transformer 用 $\\phi(x) = \\text{elu}(x) + 1$。""")

code("""class StandardAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = Q @ K.transpose(-2, -1) / np.sqrt(D)
        attn = F.softmax(scores, dim=-1)
        return attn @ V

class LinearAttention(nn.Module):
    \"\"\"线性注意力: 用 φ(x)=elu(x)+1 近似 softmax\"\"\"
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        # 核函数 φ(x) = elu(x) + 1 (保证非负)
        phi_Q = F.elu(Q) + 1
        phi_K = F.elu(K) + 1
        # 关键: 先算 K^T V → (B, D, D), 与 T 无关!
        KV = phi_K.transpose(-2, -1) @ V  # (B, D, D)
        numerator = phi_Q @ KV  # (B, T, D)
        # 归一化
        K_sum = phi_K.sum(dim=1, keepdim=True).transpose(-2, -1)  # (B, D, 1)
        denominator = (phi_Q @ K_sum) + 1e-6  # (B, T, 1)
        return numerator / denominator

# 复杂度对比: 测量不同序列长度的时间
d_model = 64
std_attn = StandardAttention(d_model)
lin_attn = LinearAttention(d_model)

seq_lengths = [64, 128, 256, 512, 1024, 2048]
timing = {'standard': [], 'linear': []}

for T in seq_lengths:
    x = torch.randn(1, T, d_model)
    # 标准
    for _ in range(3): _ = std_attn(x)
    start = time.time()
    for _ in range(20):
        with torch.no_grad(): _ = std_attn(x)
    timing['standard'].append((time.time() - start) / 20 * 1000)
    # 线性
    for _ in range(3): _ = lin_attn(x)
    start = time.time()
    for _ in range(20):
        with torch.no_grad(): _ = lin_attn(x)
    timing['linear'].append((time.time() - start) / 20 * 1000)
    print(f"T={T:5d}: 标准={timing['standard'][-1]:7.2f}ms, 线性={timing['linear'][-1]:7.2f}ms")""")

code("""# 可视化复杂度对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(seq_lengths, timing['standard'], 'r-o', linewidth=2.5, markersize=7, label='标准注意力 O(T²)')
ax.plot(seq_lengths, timing['linear'], 'b-s', linewidth=2.5, markersize=7, label='线性注意力 O(T·d²)')
ax.set_xlabel('序列长度 T')
ax.set_ylabel('时间 (ms)')
ax.set_title('标准 vs 线性注意力: 复杂度对比')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_linear_attn.png', bbox_inches='tight')
plt.show()
print("T越大, 标准注意力的 O(T²) 增长越快; 线性注意力保持线性增长。")""")

# ============================================================
md("""## 4. SSM / Mamba：状态空间模型

### 4.1 状态空间模型的基础

SSM 用一个**线性状态方程**建模序列：

$$h_t = A h_{t-1} + B x_t, \\quad y_t = C h_t$$

这看起来像 RNN！但关键区别：$A, B, C$ 在**连续时间**上定义，通过**零阶保持（ZOH）**离散化：

$$\\bar{A} = e^{\\Delta A}, \\quad \\bar{B} = (\\Delta A)^{-1}(e^{\\Delta A} - I) \\Delta B$$

### 4.2 Mamba 的创新：选择性机制

原始 SSM 的 $A, B, C$ 是**固定的**——不管输入是什么都一样。
Mamba 让它们**依赖输入**：

$$h_t = \\bar{A}(x_t) h_{t-1} + \\bar{B}(x_t) x_t$$

这样模型能**选择性地**记住或遗忘信息——像 LSTM 的门控，但线性所以能并行训练。

### 4.3 Mamba vs Transformer

| 特性 | Transformer | Mamba |
|------|-------------|-------|
| 复杂度 | $O(T^2)$ | $O(T)$ |
| 并行训练 | ✅ | ✅ (通过扫描算法) |
| 长序列 | ❌ 注意力爆炸 | ✅ 线性 |
| 全局视野 | ✅ 任意两位置直连 | ⚠️ 通过状态间接 |

> 💡 Mamba 在长序列（DNA、音频、长文本）上有优势，但 Transformer 在短序列和需要全局注意力交互的任务上仍然更强。""")

code("""class SimpleSSM(nn.Module):
    \"\"\"简化状态空间模型 (SSM)\"\"\"
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # 连续时间参数 A, B, C
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_state, d_model) * 0.1)
        # 时间步长 Δ (可学习)
        self.log_delta = nn.Parameter(torch.zeros(1))

    def discretize(self):
        \"\"\"ZOH 离散化: A_bar = exp(ΔA), B_bar = (ΔA)^{-1}(exp(ΔA) - I)ΔB\"\"\"
        delta = torch.exp(self.log_delta)
        delta_A = delta * self.A  # (d_state, d_state)
        A_bar = torch.matrix_exp(delta_A)  # exp(ΔA)
        # 简化: B_bar ≈ ΔB (一阶近似)
        B_bar = delta * self.B  # (d_model, d_state)
        return A_bar, B_bar

    def forward(self, x):
        \"\"\"x: (B, T, d_model) → (B, T, d_model)\"\"\"
        batch, T, D = x.shape
        A_bar, B_bar = self.discretize()
        # 递推: h_t = A_bar @ h_{t-1} + B_bar @ x_t
        h = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        for t in range(T):
            h = h @ A_bar + x[:, t] @ B_bar  # (batch, d_state)
            y = h @ self.C  # (batch, d_model)
            outputs.append(y)
        return torch.stack(outputs, dim=1)

class SimpleMamba(nn.Module):
    \"\"\"简化 Mamba: 选择性 SSM (A,B 依赖输入)\"\"\"
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # 输入依赖的参数生成器
        self.delta_proj = nn.Linear(d_model, 1)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        # A 用对角矩阵参数化 (简化)
        self.A_log = nn.Parameter(torch.randn(d_state) * 0.01)

    def forward(self, x):
        batch, T, D = x.shape
        outputs = []
        h = torch.zeros(batch, self.d_state, device=x.device)
        for t in range(T):
            xt = x[:, t]  # (batch, d_model)
            # 选择性参数: 依赖输入
            delta = torch.sigmoid(self.delta_proj(xt))  # (batch, 1)
            A_bar = torch.exp(-delta * torch.exp(self.A_log))  # (batch, d_state)
            B_bar = delta * self.B_proj(xt)  # (batch, d_state)
            # 状态更新 (逐元素, 因为A是对角)
            h = A_bar * h + B_bar * xt.mean(dim=-1, keepdim=True)  # (batch, d_state)
            C = self.C_proj(xt)  # (batch, d_state)
            y = (h * C).sum(dim=-1, keepdim=True)  # (batch, 1)
            outputs.append(y)
        return torch.stack(outputs, dim=1).expand(-1, -1, D)  # 广播回 d_model

# 演示
d_model = 32
ssm = SimpleSSM(d_model, d_state=16)
mamba = SimpleMamba(d_model, d_state=16)
x = torch.randn(2, 20, d_model)
print(f"SSM 输出: {ssm(x).shape}")
print(f"Mamba 输出: {mamba(x).shape}")
print("SSM/Mamba: 线性复杂度 O(T), 适合长序列。")""")

code("""# 复杂度对比: Transformer vs Mamba 在长序列上
seq_lengths = [128, 256, 512, 1024, 2048, 4096]
d_model = 32

# 用简单注意力做对比
class TinyTransformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, 3*d)
        self.d = d
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)
        attn = F.softmax(Q @ K.transpose(-2, -1) / np.sqrt(D), dim=-1)
        return attn @ V

transformer = TinyTransformer(d_model)
mamba_cmp = SimpleMamba(d_model, d_state=16)

timing = {'transformer': [], 'mamba': []}
for T in seq_lengths:
    x = torch.randn(1, T, d_model)
    # Transformer
    start = time.time()
    for _ in range(10):
        with torch.no_grad(): _ = transformer(x)
    timing['transformer'].append((time.time() - start) / 10 * 1000)
    # Mamba
    start = time.time()
    for _ in range(10):
        with torch.no_grad(): _ = mamba_cmp(x)
    timing['mamba'].append((time.time() - start) / 10 * 1000)
    print(f"T={T:5d}: Transformer={timing['transformer'][-1]:7.2f}ms, Mamba={timing['mamba'][-1]:7.2f}ms")""")

code("""fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(seq_lengths, timing['transformer'], 'r-o', linewidth=2.5, markersize=7, label='Transformer O(T²)')
ax.plot(seq_lengths, timing['mamba'], 'b-s', linewidth=2.5, markersize=7, label='Mamba O(T)')
ax.set_xlabel('序列长度 T')
ax.set_ylabel('时间 (ms)')
ax.set_title('Transformer vs Mamba: 长序列复杂度')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_mamba_vs_transformer.png', bbox_inches='tight')
plt.show()
print("注意: 教学版 Mamba 用 Python 循环实现, 开销大; 真实 Mamba 用 CUDA 并行扫描, 才能体现 O(T) 优势。")
print("理论: Transformer O(T²) vs Mamba O(T); 长序列时 Mamba 的渐近优势才显现。")""")

# ============================================================
md("""## 5. 架构创新全景

### 5.1 四类创新总结

```
                    注意力效率改进                    非注意力替代
                    ←──────────────→                  ←──────────→
         GQA/MQA          Sliding Window        线性注意力        SSM/Mamba
         (共享KV头)       (局部注意力)           (核技巧)         (状态空间)
         
         减少KV cache     减少计算量             O(T²)→O(T)      O(T²)→O(T)
         质量几乎不降      丢失全局视野           质量略降         选择性记忆
```

### 5.2 现代 LLM 用了什么？

| 模型 | 架构创新 |
|------|---------|
| **Llama 2/3** | GQA (8组) |
| **Mixtral** | MoE (8专家选2) + GQA |
| **DeepSeek-V3** | MoE (256专家) + MLA (多头潜在注意力) |
| **Mamba** | SSM + 选择性机制 |
| **Jamba** | Transformer + Mamba 混合层 |
| **Gemma 2** | GQA + Sliding Window + 软帽注意力 |

### 5.3 核心张力

> **全局视野 vs 线性复杂度**——Transformer 要 $O(T^2)$ 买全局视野，Mamba 用 $O(T)$ 但视野受限。
> 未来趋势是**混合架构**（如 Jamba）：有些层用注意力（全局交互），有些层用 SSM（高效传递）。

> 🔗 **板块一完结**：01-08 覆盖了从 MLP 到现代架构创新的完整演进。
> 下一板块进入**现代训练范式**——怎么把这些架构训练成强大的模型。""")

code("""# 架构创新全景图
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')

# 四个象限
categories = [
    (2.5, 6, '注意力效率', 'GQA / MQA\\nSliding Window\\nFlash Attention', 'lightblue'),
    (7.5, 6, '非注意力替代', '线性注意力\\nSSM / Mamba\\nRWKV', 'lightyellow'),
    (2.5, 2.5, '稀疏激活', 'MoE\\nTop-k 路由\\n负载均衡', 'lightgreen'),
    (7.5, 2.5, '混合架构', 'Jamba (Attn+SSM)\\n Griffin\\nHybrid', 'lightcoral'),
]
for x, y, title, content, color in categories:
    circle = plt.Circle((x, y), 1.8, color=color, alpha=0.7)
    ax.add_patch(circle)
    ax.text(x, y + 0.5, title, ha='center', fontsize=13, fontweight='bold')
    ax.text(x, y - 0.3, content, ha='center', fontsize=10, va='center')

ax.annotate('', xy=(5, 6), xytext=(5, 4.5),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
ax.text(5, 5.2, '效率 ↔ 质量', ha='center', fontsize=11, color='gray')

ax.set_title('现代架构创新全景', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('notebooks/fig_architecture_landscape.png', bbox_inches='tight')
plt.show()
print("架构创新围绕 效率↔质量 的张力展开——没有免费午餐。")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| GQA/MQA 共享 KV 头减少 cache | ✅ |
| MoE 稀疏激活 + top-k 路由 + 负载均衡 | ✅ |
| 线性注意力核技巧 O(T²)→O(T) | ✅ |
| SSM 状态空间模型 + ZOH 离散化 | ✅ |
| Mamba 选择性机制 | ✅ |
| 架构创新的核心张力：效率 vs 质量 | ✅ |

### 核心 takeaway

> **Transformer 的 O(T²) 是所有架构创新的靶子**。
> GQA 减 cache、MoE 减计算、线性注意力/Mamba 减复杂度——都在想办法让序列建模更高效。
> 但**没有免费午餐**：效率提升通常伴随质量下降，混合架构是未来的方向。

### 🔗 下一板块预告

**`09_tokenization.ipynb`** — BPE/SentencePiece/Unigram/tiktoken（进入现代训练范式板块）

---

> 💬 **板块一完结**：从 MLP 到现代架构，我们走完了深度学习的**基础与范式演进**。
> 接下来进入**现代训练范式**——怎么把这些架构训练成强大的模型。""")

# ============================================================
output_path = "notebooks/08_architecture_innovations.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")