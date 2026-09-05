"""生成 11_sft_peft.ipynb 的脚本"""
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
md("""# 11 — SFT + LoRA/QLoRA + DoRA/PiSSA：参数高效微调

> 预训练模型有 70B 参数，全量微调要几百 GB 显存。
> LoRA 的洞察：**微调的权重变化是低秩的**——只需更新极少量参数。
> 这让 70B 模型能在单卡 24GB 显存上微调。

## 本章你将掌握

1. **SFT**：监督微调的基本流程
2. **LoRA**：低秩分解 $\\Delta W = BA$
3. **QLoRA**：4bit 量化 + LoRA
4. **DoRA**：权重分解为方向 + 幅度
5. **对比**：全量 vs LoRA vs DoRA 的参数量和效果""")

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
md("""## 1. SFT：监督微调

### 1.1 预训练 → 微调的两阶段

```
阶段1 预训练:  海量无标注文本 → CLM/MLM → 通用模型
阶段2 SFT:     少量标注数据 (指令→回答) → 监督学习 → 任务模型
```

SFT 就是用**标注的指令数据**做监督训练：
- 输入：用户指令（"把这段翻译成英文"）
- 目标：期望回答
- 损失：CLM 损失（只在回答部分算）

### 1.2 全量微调的问题

全量微调更新**所有参数**：

| 模型大小 | 参数量 | 全量微调显存 |
|---------|--------|------------|
| 7B | 7B | ~60GB |
| 70B | 70B | ~600GB |

> 💡 70B 全量微调要 8 张 A100——绝大多数人用不起。这就是 LoRA 出现的动机。""")

code("""# 演示全量微调 vs LoRA 的参数量
class SimpleLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.W(x)

# 模拟一个 4096x4096 的权重 (类似 Llama-7B 的一层)
in_dim, out_dim = 4096, 4096
full_model = SimpleLinear(in_dim, out_dim)
full_params = sum(p.numel() for p in full_model.parameters())
print(f"全量参数: {full_params:,} ({full_params/1e6:.1f}M)")
print(f"全量微调: 需要存储 {full_params:,} 个梯度 + {full_params:,} 个优化器状态 (Adam)")
print(f"  → 约 {3 * full_params * 4 / 1e9:.1f} GB 显存 (fp32)")""")

# ============================================================
md("""## 2. LoRA：低秩适配

### 2.1 LoRA 的核心思想

LoRA 假设微调的权重变化 $\\Delta W$ 是**低秩**的，把它分解成两个小矩阵：

$$W_{\\text{new}} = W_0 + \\Delta W = W_0 + B A$$

其中 $W_0 \\in \\mathbb{R}^{d \\times d}$（冻结），$B \\in \\mathbb{R}^{d \\times r}$，$A \\in \\mathbb{R}^{r \\times d}$，$r \\ll d$。

```
全量:  ΔW (d×d) = 16M 参数        LoRA:  ΔW = B(d×r) × A(r×d) = 2×d×r 参数
      [████████]                         [██]   [████████]
      [████████]                         [██]
      [████████]                         [██]
      d×d                                d×r    r×d
```

### 2.2 参数量对比

| | 全量 | LoRA (r=8) |
|---|------|-----------|
| 参数量 | $d^2$ | $2dr$ |
| 比例 | 100% | $2r/d$ |
| d=4096, r=8 | 16M | 65K (0.4%) |

> 💡 LoRA 只更新 0.4% 的参数，效果接近全量微调！这是 2021 年微软的重磅发现。""")

code("""class LoRALinear(nn.Module):
    \"\"\"LoRA: W_new = W0 + B*A, 只训练 B 和 A\"\"\"
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        # 原始权重 (冻结)
        self.W0 = nn.Linear(in_dim, out_dim, bias=False)
        self.W0.weight.requires_grad = False
        # LoRA 分解: ΔW = B @ A
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)   # (r, in)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))          # (out, r), 初始0
        self.scaling = alpha / rank  # 缩放因子

    def forward(self, x):
        # W0 @ x + scaling * B @ A @ x
        return self.W0(x) + self.scaling * (x @ self.A.T @ self.B.T)

# 对比参数量
rank = 8
lora_model = LoRALinear(in_dim, out_dim, rank=rank)
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in lora_model.parameters() if not p.requires_grad)

print(f"LoRA (rank={rank}):")
print(f"  可训练参数: {trainable:,} ({trainable/full_params:.2%} of full)")
print(f"  冻结参数: {frozen:,}")
print(f"  显存: 只需 {trainable * 3 * 4 / 1e6:.1f} MB (参数+梯度+优化器状态)")
print(f"  vs 全量: {full_params * 3 * 4 / 1e9:.1f} GB")""")

code("""# 不同 rank 的参数量对比
ranks = [1, 2, 4, 8, 16, 32, 64, 128]
params = [2 * in_dim * r for r in ranks]
percentages = [p / full_params * 100 for p in params]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(ranks)), percentages, color='steelblue', alpha=0.7)
ax.set_xticks(range(len(ranks)))
ax.set_xticklabels([f'r={r}' for r in ranks])
ax.set_ylabel('可训练参数占比 (%)')
ax.set_title(f'LoRA: 不同 rank 的参数量 (d={in_dim})')
ax.grid(True, alpha=0.3, axis='y')
for bar, pct in zip(bars, percentages):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{pct:.2f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('notebooks/fig_lora_ranks.png', bbox_inches='tight')
plt.show()
print("rank=8 时只更新 0.39% 参数——效果却接近全量微调。")""")

# ============================================================
md("""## 3. LoRA 的训练过程

### 3.1 初始化技巧

LoRA 的初始化很巧妙：
- $A$ 用高斯随机初始化
- $B$ **初始化为 0**

这样训练开始时 $\\Delta W = B \\cdot A = 0$，模型等于预训练模型——**从预训练状态出发**。

### 3.2 缩放因子 $\\alpha/r$

$$h = W_0 x + \\frac{\\alpha}{r} B A x$$

$\\alpha$ 控制 LoRA 更新的"学习率"。常见设置：$\\alpha = 2r$ 或 $\\alpha = 16$。""")

code("""# LoRA 微调实验: 迁移学习场景
# 预训练: sin(x), 微调目标: sin(x+1) (相位偏移)
torch.manual_seed(42)

n_samples = 200
X_train = torch.randn(n_samples, 1) * 3
y_pretrain = torch.sin(X_train) + 0.1 * torch.randn(n_samples, 1)   # 预训练目标
y_finetune = torch.sin(X_train + 1.0) + 0.1 * torch.randn(n_samples, 1)  # 微调目标

class BaseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

# "预训练"基础模型在 sin(x) 上
base_model = BaseMLP()
optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-2)
for _ in range(300):
    optimizer.zero_grad()
    loss = F.mse_loss(base_model(X_train), y_pretrain)
    loss.backward()
    optimizer.step()

# 评估预训练模型在微调任务上的表现
with torch.no_grad():
    pre_ft_loss = F.mse_loss(base_model(X_train), y_finetune).item()
print(f"预训练损失(sin(x)): {loss.item():.4f}")
print(f"预训练模型在 sin(x+1) 上的损失: {pre_ft_loss:.4f} (需要微调)")

# 全量微调到 sin(x+1)
full_ft = BaseMLP()
full_ft.load_state_dict(base_model.state_dict())
optimizer = torch.optim.Adam(full_ft.parameters(), lr=1e-3)
losses_full = []
for _ in range(300):
    optimizer.zero_grad()
    loss = F.mse_loss(full_ft(X_train), y_finetune)
    loss.backward()
    optimizer.step()
    losses_full.append(loss.item())

# LoRA 微调 (给每层加 LoRA)
class LoRAMLP(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.fc1 = LoRALinear(1, 64, rank=rank)
        self.fc2 = LoRALinear(64, 64, rank=rank)
        self.fc3 = LoRALinear(64, 1, rank=rank)
        self.fc1.W0.weight.data = base.fc1.weight.data.clone()
        self.fc2.W0.weight.data = base.fc2.weight.data.clone()
        self.fc3.W0.weight.data = base.fc3.weight.data.clone()
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

lora_ft = LoRAMLP(base_model, rank=16)
lora_params = sum(p.numel() for p in lora_ft.parameters() if p.requires_grad)
full_params_mlp = sum(p.numel() for p in full_ft.parameters())
print(f"\\n全量微调参数: {full_params_mlp:,}")
print(f"LoRA 微调参数: {lora_params:,} ({lora_params/full_params_mlp:.1%})")

optimizer = torch.optim.Adam([p for p in lora_ft.parameters() if p.requires_grad], lr=1e-2)
losses_lora = []
for _ in range(500):
    optimizer.zero_grad()
    loss = F.mse_loss(lora_ft(X_train), y_finetune)
    loss.backward()
    optimizer.step()
    losses_lora.append(loss.item())

print(f"全量微调最终损失: {losses_full[-1]:.4f}")
print(f"LoRA 微调最终损失: {losses_lora[-1]:.4f}")""")

code("""# 可视化训练曲线
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(losses_full, 'b-', linewidth=2, label='全量微调')
axes[0].plot(losses_lora, 'r-', linewidth=2, label='LoRA (r=16)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('训练损失对比'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 可视化拟合结果
X_plot = torch.linspace(-6, 6, 200).reshape(-1, 1)
with torch.no_grad():
    y_base = base_model(X_plot)
    y_full = full_ft(X_plot)
    y_lora = lora_ft(X_plot)

axes[1].scatter(X_train.numpy(), y_finetune.numpy(), s=5, alpha=0.3, color='gray', label='微调数据')
axes[1].plot(X_plot.numpy(), torch.sin(X_plot + 1.0).numpy(), 'k--', linewidth=2, label='真实 sin(x+1)')
axes[1].plot(X_plot.numpy(), y_full.numpy(), 'b-', linewidth=2, label='全量微调')
axes[1].plot(X_plot.numpy(), y_lora.numpy(), 'r-', linewidth=2, label='LoRA')
axes[1].set_title('拟合结果'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_lora_vs_full.png', bbox_inches='tight')
plt.show()
print("LoRA 用极少参数达到接近全量微调的效果——这就是它的威力。")""")

# ============================================================
md("""## 4. QLoRA：4bit 量化 + LoRA

### 4.1 QLoRA 的三重优化

QLoRA 把 LoRA 推到极致——**冻结的 $W_0$ 用 4bit 存储**：

| 组件 | 全量微调 | LoRA | QLoRA |
|------|---------|------|-------|
| $W_0$ | fp32 (4B/参数) | fp32 (4B) | **4bit (0.5B)** |
| $\\Delta W = BA$ | - | fp32 | fp16 |
| 70B 显存 | ~600GB | ~160GB | **~40GB** |

### 4.2 NormalFloat 4bit (NF4)

QLoRA 用特殊的 **NF4** 量化——假设权重服从正态分布，用信息论最优的 16 个量化点：

```
fp32:  -0.123, -0.087, -0.041, 0.012, 0.056, 0.098, ...
NF4:   -0.125, -0.083, -0.042, 0.000, 0.052, 0.094, ...  (16个固定点)
```

> 💡 QLoRA 让 70B 模型在**单张 48GB A6000** 上微调—— democratize 了 LLM 微调。""")

code("""# 简化 4bit 量化演示
def quantize_4bit(weight):
    \"\"\"简化 4bit 量化: 把 fp32 权重量化到 16 个级别\"\"\"
    # NF4 量化点 (简化的正态分布最优量化)
    nf4_levels = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3947, -0.2849, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])
    # 归一化权重到 [-1, 1]
    w_max = weight.abs().max()
    w_norm = weight / (w_max + 1e-8)
    # 量化: 找最近的级别
    indices = torch.argmin((w_norm.unsqueeze(-1) - nf4_levels) ** 2, dim=-1)
    # 反量化
    w_dequant = nf4_levels[indices] * w_max
    return w_dequant, indices

# 演示量化
W = torch.randn(64, 64) * 0.1
W_dequant, indices = quantize_4bit(W)
quant_error = (W - W_dequant).norm() / W.norm()

print(f"原始: {W.numel() * 4 / 1024:.1f} KB (fp32)")
print(f"量化后: {indices.numel() * 0.5 / 1024:.1f} KB (4bit) — 压缩 8x")
print(f"量化误差: {quant_error:.4f} (相对范数)")

# 可视化量化
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
W_small = torch.randn(20, 20) * 0.1
W_dq, _ = quantize_4bit(W_small)

vmin, vmax = W_small.min(), W_small.max()
axes[0].imshow(W_small.numpy(), cmap='RdBu', vmin=vmin, vmax=vmax)
axes[0].set_title('原始权重 (fp32)')
axes[1].imshow(W_dq.numpy(), cmap='RdBu', vmin=vmin, vmax=vmax)
axes[1].set_title('4bit 量化后')
axes[2].imshow((W_small - W_dq).numpy(), cmap='RdBu', vmin=-0.05, vmax=0.05)
axes[2].set_title('量化误差')
plt.tight_layout()
plt.savefig('notebooks/fig_4bit_quant.png', bbox_inches='tight')
plt.show()
print("4bit 量化压缩 8x, 误差很小——QLoRA 的基础。")""")

# ============================================================
md("""## 5. DoRA：权重分解微调

### 5.1 DoRA 的改进

DoRA（2024）把权重分解为**方向**和**幅度**：

$$W = m \\cdot \\frac{V}{\\|V\\|}$$

其中 $m$ 是幅度（每列一个标量），$V/\\|V\\|$ 是方向。LoRA 只更新方向，DoRA **同时更新方向和幅度**：

```
LoRA:  W_new = W0 + BA         (只调方向)
DoRA:  W_new = m_new · V_new/|V_new|  (调方向 + 调幅度)
```

### 5.2 为什么 DoRA 更好？

LoRA 的 $\\Delta W = BA$ 倾向于同时改变方向和幅度，但**幅度变化不够灵活**。
DoRA 解耦后能分别控制，更接近全量微调的行为。

> 💡 DoRA 在 rank=8 时效果接近 rank=16 的 LoRA——**同等参数量下更好**。""")

code("""class DoRALinear(nn.Module):
    \"\"\"DoRA: W = m * V/||V||, LoRA更新V, m也可学习\"\"\"
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.W0 = nn.Linear(in_dim, out_dim, bias=False)
        self.W0.weight.requires_grad = False
        # LoRA 部分
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scaling = alpha / rank
        # DoRA: 可学习的幅度向量 (每列一个)
        # 初始化为 W0 每列的范数
        with torch.no_grad():
            m_init = self.W0.weight.norm(dim=0, keepdim=True)
        self.m = nn.Parameter(m_init.clone())

    def forward(self, x):
        # V = W0 + scaling * B @ A
        V = self.W0.weight + self.scaling * (self.B @ self.A)
        # 方向归一化
        V_norm = V / (V.norm(dim=0, keepdim=True) + 1e-8)
        # W = m * V_norm
        W = self.m * V_norm
        return x @ W.T

# 对比全量 vs LoRA vs DoRA
torch.manual_seed(42)
base_model2 = BaseMLP()
optimizer = torch.optim.Adam(base_model2.parameters(), lr=1e-2)
for _ in range(300):
    optimizer.zero_grad()
    loss = F.mse_loss(base_model2(X_train), y_pretrain)
    loss.backward()
    optimizer.step()

# DoRA 微调
class DoRAMLP(nn.Module):
    def __init__(self, base, rank=4):
        super().__init__()
        self.fc1 = DoRALinear(1, 64, rank=rank)
        self.fc2 = DoRALinear(64, 64, rank=rank)
        self.fc3 = DoRALinear(64, 1, rank=rank)
        self.fc1.W0.weight.data = base.fc1.weight.data.clone()
        self.fc2.W0.weight.data = base.fc2.weight.data.clone()
        self.fc3.W0.weight.data = base.fc3.weight.data.clone()
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

dora_ft = DoRAMLP(base_model2, rank=16)
dora_params = sum(p.numel() for p in dora_ft.parameters() if p.requires_grad)
print(f"DoRA 参数: {dora_params:,} (vs LoRA: {lora_params:,})")

optimizer = torch.optim.Adam([p for p in dora_ft.parameters() if p.requires_grad], lr=1e-2)
losses_dora = []
for _ in range(500):
    optimizer.zero_grad()
    loss = F.mse_loss(dora_ft(X_train), y_finetune)
    loss.backward()
    optimizer.step()
    losses_dora.append(loss.item())

print(f"全量微调损失: {losses_full[-1]:.4f}")
print(f"LoRA 微调损失: {losses_lora[-1]:.4f}")
print(f"DoRA 微调损失: {losses_dora[-1]:.4f}")""")

code("""# 可视化三种方法对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(losses_full, 'b-', linewidth=2.5, label='全量微调')
ax.plot(losses_lora, 'r-', linewidth=2.5, label='LoRA (r=4)')
ax.plot(losses_dora, 'g-', linewidth=2.5, label='DoRA (r=16)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title('全量 vs LoRA vs DoRA: 训练损失对比')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_dora_comparison.png', bbox_inches='tight')
plt.show()
print("DoRA 通常比同 rank 的 LoRA 更接近全量微调——解耦方向和幅度的优势。")""")

# ============================================================
md("""## 6. 微调方法全景

### 6.1 对比总结

| 方法 | 可训练参数 | 显存 (70B) | 效果 | 适用场景 |
|------|-----------|-----------|------|---------|
| **全量微调** | 100% | ~600GB | 最好 | 研究院/大集群 |
| **LoRA** | ~0.4% | ~160GB | 接近全量 | 主流选择 |
| **QLoRA** | ~0.4% | ~40GB | 略降 | 单卡微调 |
| **DoRA** | ~0.5% | ~160GB | 优于 LoRA | 新选择 |

### 6.2 选择指南

```
有 8+ A100?     → 全量微调 (效果最好)
有 1-2 A100?    → LoRA (性价比最高)
只有 1 张消费级? → QLoRA (能跑就行)
追求最优效果?   → DoRA (同参数量更好)
```

> 💡 实际中 LoRA/QLoRA 是**绝大多数开发者的选择**——效果接近全量，资源需求低 10-100x。""")

code("""# 参数量和显存对比图
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

methods = ['全量微调', 'LoRA', 'QLoRA', 'DoRA']
param_pct = [100, 0.4, 0.4, 0.5]
vram_gb = [600, 160, 40, 160]

colors = ['red', 'blue', 'green', 'orange']
axes[0].bar(methods, param_pct, color=colors, alpha=0.7)
axes[0].set_ylabel('可训练参数占比 (%)')
axes[0].set_title('参数量对比 (70B 模型)')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(methods, vram_gb, color=colors, alpha=0.7)
axes[1].set_ylabel('显存需求 (GB)')
axes[1].set_title('显存对比 (70B 模型)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/fig_peft_comparison.png', bbox_inches='tight')
plt.show()
print("QLoRA 把 70B 微调从 600GB 降到 40GB——让单卡微调成为可能。")""")

# ============================================================
md("""## 7. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| SFT 监督微调的两阶段流程 | ✅ |
| LoRA 低秩分解 ΔW = BA | ✅ |
| LoRA 初始化技巧 (B=0, A随机) | ✅ |
| QLoRA 4bit 量化 + LoRA | ✅ |
| DoRA 权重分解方向+幅度 | ✅ |
| 不同方法的参数量/显存/效果对比 | ✅ |

### 核心 takeaway

> **LoRA 让大模型微调民主化**——0.4% 参数接近全量效果，QLoRA 进一步降到单卡可跑。
> DoRA 解耦方向和幅度，同参数量下更优。现代微调首选 LoRA/QLoRA。

### 🔗 下一章预告

**`12_rlhf_ppo.ipynb`** — 🧭RLHF + PPO 底层机制（强化学习对齐）

---

> 💬 **写在最后**：预训练是"通识教育"，SFT 是"专业训练"，LoRA 是"高效专业训练"。
> 理解 LoRA 的低秩假设，就理解了为什么大模型能高效适配下游任务。""")

# ============================================================
output_path = "notebooks/11_sft_peft.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")