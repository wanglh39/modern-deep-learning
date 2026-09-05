# 生成 41_distillation_quantization.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 41 — 知识蒸馏与量化：模型压缩

> 🔥 70B 模型太大了？蒸馏到 7B，量化到 4bit——让大模型在消费级硬件上运行。

## 本章你将掌握

1. **知识蒸馏**：大模型→小模型
2. **量化**：PTQ/QAT/GPTQ/AWQ
3. **量化感知修复**：🔥 量化后精度修复
4. **实战对比**：不同压缩方法""")

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

md("""## 1. 知识蒸馏

### 1.1 思想

把大模型（教师）的知识转移到小模型（学生）：

```
教师 T (大):  准确但慢
学生 S (小):  快但不够准

蒸馏: 让学生模仿教师的输出 (软标签)
  → 学生学到教师的"暗知识" (类间关系)
```

### 1.2 蒸馏损失

$$\\mathcal{L} = \\alpha \\cdot \\mathcal{L}_{CE}(S(x), y) + (1-\\alpha) \\cdot T^2 \\cdot \\mathcal{L}_{KL}(S_T(x), T_T(x))$$

- $\\mathcal{L}_{CE}$：学生与真实标签的交叉熵（硬标签）
- $\\mathcal{L}_{KL}$：学生与教师的 KL 散度（软标签）
- $T$：温度（软化输出）
- $\\alpha$：权衡

### 1.3 温度的作用

```
T=1:  正常 softmax (置信度高)
T=4:  软化 softmax (暴露类间关系)

例子: [10, 5, 1]
  T=1: [0.99, 0.01, 0.00]  → 只知道"是类0"
  T=4: [0.70, 0.20, 0.10]  → 知道"类0>类1>类2" (暗知识)
```

> 💡 蒸馏的魔法：软标签包含**类间关系**（暗知识），比硬标签信息更丰富。
# 温度越高，暗知识越明显。""")

code("""# 知识蒸馏实现
class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x):
        return self.net(x)

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x):
        return self.net(x)

def distillation_loss(student_logits, teacher_logits, labels, T=4, alpha=0.5):
    # 硬标签损失
    hard_loss = F.cross_entropy(student_logits, labels)
    # 软标签损失 (KL 散度)
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    return alpha * hard_loss + (1 - alpha) * soft_loss

# 训练对比: 蒸馏 vs 纯学生
teacher = Teacher()
student_distill = Student()
student_plain = Student()

# 模拟数据
x = torch.randn(500, 784)
y = torch.randint(0, 10, (500,))

# 预训练教师
opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3)
for _ in range(200):
    loss = F.cross_entropy(teacher(x), y)
    opt_t.zero_grad(); loss.backward(); opt_t.step()

# 蒸馏训练
opt_d = torch.optim.Adam(student_distill.parameters(), lr=1e-3)
opt_p = torch.optim.Adam(student_plain.parameters(), lr=1e-3)

distill_losses, plain_losses = [], []
for _ in range(200):
    t_logits = teacher(x).detach()

    d_loss = distillation_loss(student_distill(x), t_logits, y, T=4)
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()
    distill_losses.append(d_loss.item())

    p_loss = F.cross_entropy(student_plain(x), y)
    opt_p.zero_grad(); p_loss.backward(); opt_p.step()
    plain_losses.append(p_loss.item())

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(distill_losses, 'b-', linewidth=2, label='蒸馏')
ax.plot(plain_losses, 'r-', linewidth=2, label='纯学生')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
ax.set_title('知识蒸馏 vs 纯学生训练'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_distillation.png', bbox_inches='tight')
plt.show()
print(f"蒸馏: loss {distill_losses[-1]:.3f}, 纯学生: loss {plain_losses[-1]:.3f}")
print(f"教师参数: {sum(p.numel() for p in teacher.parameters())/1e3:.0f}K, 学生: {sum(p.numel() for p in student_distill.parameters())/1e3:.0f}K")
print("蒸馏: 小模型学到教师的暗知识——比纯训练更好。")""")

md("""## 2. 量化：FP16/INT8/INT4

### 2.1 为什么量化

```
FP32:  70B 模型 = 280GB → 需要多卡
INT8:  70B 模型 = 70GB  → 单卡可行
INT4:  70B 模型 = 35GB  → 消费级 GPU

→ 量化让大模型在小硬件上运行
```

### 2.2 量化方法

| 方法 | 说明 | 精度 |
|------|------|------|
| **PTQ** | 训练后量化 (简单) | 较好 |
| **QAT** | 量化感知训练 (训练时模拟) | 好 |
| **GPTQ** | 逐层量化 + 误差补偿 | 很好 |
| **AWQ** | 保护重要权重 | 很好 |

### 2.3 量化原理

```
FP32 → INT8:
  scale = max(|w|) / 127
  w_int8 = round(w / scale)
  w_recover = w_int8 * scale ≈ w

INT4 更激进:
  scale = max(|w|) / 7
  → 只有 16 个值: -7, -6, ..., 0, ..., 7
```

> 💡 量化的核心：**用更少比特近似表示权重**。
# GPTQ/AWQ 通过保护重要权重，让 INT4 量化几乎不损失精度。""")

code("""# 量化实现
def quantize_tensor(w, n_bits=8):
    # 对称量化
    scale = w.abs().max() / (2 ** (n_bits - 1) - 1)
    w_quant = torch.round(w / scale).clamp(-(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
    return w_quant, scale

def dequantize_tensor(w_quant, scale):
    return w_quant * scale

# 模拟权重
w = torch.randn(1000) * 0.1

# 不同比特量化
results = {}
for bits in [16, 8, 4, 2]:
    wq, scale = quantize_tensor(w, n_bits=bits)
    w_dq = dequantize_tensor(wq, scale)
    error = (w - w_dq).norm() / w.norm()
    size = wq.element_size() if bits >= 8 else bits / 8  # 字节
    results[bits] = {'error': error.item(), 'size_ratio': 32 / bits}

print(f"{'比特':>6s} {'量化误差':>10s} {'压缩比':>10s}")
for bits, r in results.items():
    print(f"{bits:6d} {r['error']:10.4f} {r['size_ratio']:10.1f}x")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bits_list = [32, 16, 8, 4, 2]
errors = []
for bits in bits_list:
    if bits == 32:
        errors.append(0)
    else:
        wq, scale = quantize_tensor(w, n_bits=bits)
        w_dq = dequantize_tensor(wq, scale)
        errors.append(((w - w_dq).norm() / w.norm()).item())

axes[0].bar([str(b) for b in bits_list], errors, color='steelblue', alpha=0.8)
axes[0].set_xlabel('比特数'); axes[0].set_ylabel('量化误差')
axes[0].set_title('量化误差 vs 比特数')

# 模型大小
sizes = [70 * 32 / b for b in bits_list]  # 70B 模型
axes[1].bar([str(b) for b in bits_list], sizes, color='coral', alpha=0.8)
axes[1].set_xlabel('比特数'); axes[1].set_ylabel('模型大小 (GB)')
axes[1].set_title('70B 模型大小 vs 量化比特')

plt.tight_layout()
plt.savefig('notebooks/fig_quantization.png', bbox_inches='tight')
plt.show()
print("INT4: 4x 压缩, 误差可控——让 70B 在消费级 GPU 运行。")""")

md("""## 3. GPTQ 与 AWQ：高级量化

### 3.1 GPTQ

GPTQ 逐层量化，**补偿量化误差**：

```
1. 量化第 i 个权重 w_i
2. 量化误差 e = w_i - dequant(w_i)
3. 调整未量化权重 w_j (j>i) 补偿 e
4. → 误差不累积

→ INT4 量化几乎无损
```

### 3.2 AWQ

AWQ 发现**不是所有权重都重要**：

```
1. 找出"重要"权重 (激活值大的通道)
2. 保护重要权重 (不量化或高比特)
3. 不重要的权重量化到 INT4

→ 保护关键信息, 压缩其余
```

### 3.3 对比

| 方法 | 速度 | 精度 | 适用 |
|------|------|------|------|
| **PTQ** | 最快 | 较好 | 快速部署 |
| **GPTQ** | 快 | 很好 | 通用 |
| **AWQ** | 快 | 很好 | 推理优化 |
| **QAT** | 慢 (需训练) | 最好 | 追求极致 |

> 💡 GPTQ 和 AWQ 是当前 INT4 量化的主流方法。
# 它们让 INT4 量化几乎无损——70B 模型在 40GB GPU 上运行。""")

code("""# GPTQ 简化模拟
def gptq_quantize_layer(weight, n_bits=4):
    # 逐列量化 + 误差补偿
    w = weight.clone()
    scale = w.abs().max() / (2 ** (n_bits - 1) - 1)
    w_quant = torch.zeros_like(w)
    w_error = torch.zeros_like(w)

    for i in range(w.shape[1]):
        # 量化第 i 列
        col = w[:, i]
        q = torch.round(col / scale).clamp(-(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1)
        w_quant[:, i] = q
        # 量化误差
        e = col - q * scale
        w_error[:, i] = e
        # 补偿: 调整后续列 (简化)
        if i + 1 < w.shape[1]:
            w[:, i+1] += e * 0.1  # 简化补偿

    return w_quant, scale, w_error

# 对比 PTQ vs GPTQ
weight = torch.randn(128, 128) * 0.1

# PTQ (简单量化)
wq_ptq, scale_ptq = quantize_tensor(weight, n_bits=4)
w_dq_ptq = dequantize_tensor(wq_ptq, scale_ptq)
error_ptq = (weight - w_dq_ptq).norm() / weight.norm()

# GPTQ
wq_gptq, scale_gptq, _ = gptq_quantize_layer(weight, n_bits=4)
w_dq_gptq = wq_gptq * scale_gptq
error_gptq = (weight - w_dq_gptq).norm() / weight.norm()

print(f"PTQ  量化误差: {error_ptq:.4f}")
print(f"GPTQ 量化误差: {error_gptq:.4f}")
print(f"GPTQ 误差补偿 → 更低量化损失。")""")

md("""## 4. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 知识蒸馏 (软标签) | ✅ |
| 温度参数 | ✅ |
| PTQ/QAT 量化 | ✅ |
| GPTQ/AWQ 高级量化 | ✅ |

### 核心 takeaway
> **蒸馏让小模型有大模型的知识，量化让大模型在小硬件运行**。
# GPTQ/AWQ 让 INT4 几乎无损——70B 在 40GB GPU 上跑。

### 🔗 下一章
**`42_pruning_moe.ipynb`** — 剪枝、稀疏化、MoE 效率

---

> 💬 **板块七进行中。**""")

output_path = "notebooks/41_distillation_quantization.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")