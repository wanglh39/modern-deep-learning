# 生成 40_autoregressive_vs_diffusion.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 40 — 自回归 vs 扩散：生成范式对比

> 自回归（GPT）和扩散（SD）是两大生成范式。
> 各有优劣，正在融合——LlamaGen 用自回归做图像，SD3 用 Flow Matching 接近自回归。

## 本章你将掌握

1. **自回归生成**：逐 token 生成
2. **扩散生成**：逐步去噪
3. **对比分析**：速度、质量、可控性
4. **融合趋势**：两种范式的统一""")

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

md("""## 1. 自回归生成

### 1.1 原理

```
P(x) = P(x1) * P(x2|x1) * P(x3|x1,x2) * ... * P(xn|x1,...,xn-1)

逐 token 生成:
  x1 ~ P(x1)
  x2 ~ P(x2|x1)
  x3 ~ P(x3|x1,x2)
  ...

代表: GPT (文本), LlamaGen (图像), VQ-VAE (图像)
```

### 1.2 优势

- **灵活**：任意长度，条件生成自然
- **可解释**：每步生成一个 token
- **指令跟随**：文本生成本身就是自回归

### 1.3 劣势

- **慢**：N 个 token 要 N 次前向
- **误差累积**：早期错误传播
- **离散**：token 是离散的，梯度不好回传

> 💡 自回归是 LLM 的基石——文本生成本质就是自回归。
# 但图像生成用自回归（如 LlamaGen）是新趋势。""")

md("""## 2. 扩散生成

### 2.1 原理

```
P(x) 通过反向扩散过程定义:
  xT ~ N(0, I)
  x_{t-1} ~ P(x_{t-1}|x_t)  (学去噪)
  ...
  x0 = 生成样本

代表: DDPM, Stable Diffusion, Sora
```

### 2.2 优势

- **质量高**：当前最好的生成质量
- **训练稳定**：MSE 损失，不像 GAN 难训
- **可控**：classifier-free guidance, ControlNet

### 2.3 劣势

- **慢**：需要多步去噪（虽然 DDIM/Flow 加速）
- **固定长度**：不像自回归可以任意长度
- **不自然做条件**：需要额外机制（cross-attention）

> 💡 扩散是图像/视频生成的王者——质量好、训练稳。
# 但速度和灵活性不如自回归。""")

code("""# 自回归 vs 扩散: 生成过程对比
def autoregressive_gen(model, n_tokens=100):
    # 自回归: 逐 token 生成
    tokens = []
    time_per_token = []
    for i in range(n_tokens):
        start = time.time()
        # 模拟一次前向
        _ = np.random.randn(768)
        t = time.time() - start
        time_per_token.append(t)
        tokens.append(np.random.randint(0, 1000))
    return tokens, time_per_token

def diffusion_gen(model, n_steps=50):
    # 扩散: 逐步去噪
    x = np.random.randn(64, 64, 4)  # 潜变量
    time_per_step = []
    for i in range(n_steps):
        start = time.time()
        # 模拟一次 U-Net 前向
        _ = np.random.randn(64, 64, 4)
        t = time.time() - start
        time_per_step.append(t)
    return x, time_per_step

import time
ar_tokens, ar_times = autoregressive_gen(None, n_tokens=256)
diff_samples, diff_times = diffusion_gen(None, n_steps=50)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 生成过程
ax = axes[0]
ax.bar(['自回归\\n(256 tokens)'], [sum(ar_times)*1000], color='steelblue', alpha=0.8)
ax.bar(['扩散\\n(50 steps)'], [sum(diff_times)*1000], color='coral', alpha=0.8)
ax.set_ylabel('生成时间 (ms)')
ax.set_title('生成速度对比')

# 质量 vs 速度
ax = axes[1]
ar_steps = [1, 10, 50, 100, 256, 512]
ar_quality = [30, 50, 70, 80, 88, 92]
diff_steps = [1, 4, 8, 20, 50, 100]
diff_quality = [60, 80, 88, 93, 95, 96]

ax.plot(ar_steps, ar_quality, 'b-o', linewidth=2, label='自回归', markersize=6)
ax.plot(diff_steps, diff_quality, 'r-s', linewidth=2, label='扩散', markersize=6)
ax.set_xlabel('步数'); ax.set_ylabel('质量 (%)')
ax.set_title('质量 vs 步数'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig('notebooks/fig_ar_vs_diff.png', bbox_inches='tight')
plt.show()
print("自回归: 逐步生成, 灵活但慢; 扩散: 逐步去噪, 质量好但固定长度。")""")

md("""## 3. 对比总结

| 维度 | 自回归 | 扩散 |
|------|--------|------|
| **生成方式** | 逐 token | 逐步去噪 |
| **速度** | N 次前向 | T 步去噪 |
| **质量** | 好 (文本) / 中 (图像) | 最好 (图像) |
| **训练** | 交叉熵 | MSE |
| **长度** | 任意 | 固定 |
| **条件** | 自然 (prompt) | 需 CFG |
| **可控性** | 中 | 高 (ControlNet) |
| **代表** | GPT, LlamaGen | SD, Sora |

### 3.1 各自的领域

```
文本生成: 自回归 (GPT) — 扩散不适合 (离散 token)
图像生成: 扩散 (SD) — 质量更好, 但 LlamaGen 挑战
视频生成: 扩散 (Sora) — 时空一致性
音频生成: 两者都有 (VALL-E 自回归, AudioLM 扩散)
```""")

code("""# 领域适用性雷达图
categories = ['文本', '图像', '视频', '音频', '3D', '代码']
ar_scores = [95, 70, 50, 80, 60, 95]  # 自回归
diff_scores = [40, 95, 90, 75, 85, 30]  # 扩散

angles = [n / len(categories) * 2 * np.pi for n in range(len(categories))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.plot(angles, ar_scores + ar_scores[:1], 'b-o', linewidth=2, label='自回归')
ax.fill(angles, ar_scores + ar_scores[:1], alpha=0.15, color='blue')
ax.plot(angles, diff_scores + diff_scores[:1], 'r-s', linewidth=2, label='扩散')
ax.fill(angles, diff_scores + diff_scores[:1], alpha=0.15, color='red')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 100)
ax.set_title('自回归 vs 扩散: 各领域适用性', fontsize=13, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('notebooks/fig_ar_diff_radar.png', bbox_inches='tight')
plt.show()
print("自回归: 文本/代码强; 扩散: 图像/视频/3D强——互补。")""")

md("""## 4. 融合趋势

### 4.1 自回归做图像

```
LlamaGen (2024):
  图像 → VQ tokenizer → 离散 token → 自回归生成
  → 用 GPT 式方法做图像生成
  → 质量接近 SD, 速度更快
```

### 4.2 扩散接近自回归

```
Flow Matching / Rectified Flow:
  扩散的连续化 → 路径更直 → 步数更少
  → 4-8 步生成 → 接近自回归的速度
```

### 4.3 统一视角

```
自回归: P(x) = Π P(xi | x<i)  (链式法则)
扩散:   P(x) 通过反向过程    (变分推断)
Flow:   P(x) 通过 ODE        (连续流)

→ 都是把复杂分布分解为简单步骤
→ 本质上在做同一件事
```

> 💡 两大范式正在融合：
# LlamaGen 用自回归做图像，SD3 用 Flow Matching 接近自回归速度。
# 未来可能不再区分——都是"分步生成"。

### 4.4 MAR (Masked Autoregressive)

```
MAR (2024):
  图像 → patch → 随机顺序掩码 → 逐步揭示
  → 结合自回归和掩码建模
  → 质量接近扩散, 速度接近自回归
```

> 💡 MAR 是融合的最新尝试——
# 用掩码建模的框架统一自回归和扩散。""")

code("""# 融合趋势时间线
fig, ax = plt.subplots(figsize=(13, 5))

events = [
    (2020, 'DDPM', '扩散', 'red'),
    (2022, 'Stable Diffusion', '扩散', 'red'),
    (2023, 'LlamaGen', '自回归→图像', 'blue'),
    (2024, 'SD3 (Flow Matching)', '扩散→快', 'red'),
    (2024, 'MAR', '融合', 'green'),
    (2024, 'Flux', '扩散→快', 'red'),
]

for year, name, desc, color in events:
    ax.scatter(year, 0, s=150, zorder=5, color=color)
    ax.annotate(f'{name}\\n({desc})', xy=(year, 0), xytext=(year, 0.3 + 0.15 * (year % 2)),
                ha='center', fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlim(2019, 2025); ax.set_ylim(-0.5, 1.2)
ax.set_xlabel('年份'); ax.set_title('自回归与扩散的融合趋势')
ax.axhline(0, color='gray', linewidth=0.5); ax.set_yticks([])
plt.tight_layout()
plt.savefig('notebooks/fig_convergence.png', bbox_inches='tight')
plt.show()
print("趋势: 自回归和扩散正在融合——都是分步生成的不同实现。")""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 自回归生成 | ✅ |
| 扩散生成 | ✅ |
| 速度/质量/可控性对比 | ✅ |
| 融合趋势 | ✅ |

### 核心 takeaway
> **自回归适合文本/代码，扩散适合图像/视频**——各有所长。
> 两大范式正在融合：LlamaGen 用自回归做图像，Flow Matching 让扩散更快。
> 本质都是"分步生成"——未来可能统一。

### 🔗 下一板块
**`41_quantization.ipynb`** — 量化、剪枝、蒸馏（进入板块七：效率与部署）

---

> 💬 **板块六(生成模型)完结。**""")

output_path = "notebooks/40_autoregressive_vs_diffusion.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")
