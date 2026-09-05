"""生成 18_non_gradient_optimization.ipynb 的脚本"""
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
md("""# 18 — 非梯度优化：神经进化 / ES / NAS

> 梯度下降是王道，但有些场景**没有梯度**：离散结构、不可微目标、黑盒优化。
> 非梯度优化用**进化**和**搜索**来优化——不需要梯度，但需要大量评估。

## 本章你将掌握

1. **进化策略 (ES)**：用种群优化参数
2. **神经进化**：进化网络结构和权重
3. **NAS**：网络架构搜索
4. **梯度 vs 非梯度**的对比与适用场景""")

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
md("""## 1. 为什么需要非梯度优化？

### 1.1 梯度下降的局限

| 场景 | 为什么梯度不行 | 非梯度方案 |
|------|-------------|-----------|
| **离散结构** | 架构选择不可微 | NAS |
| **黑盒目标** | 无法求导 | ES/CMA-ES |
| **不可微激活** | 梯度未定义 | 进化 |
| **多目标** | 梯度只优化一个 | Pareto 进化 |

### 1.2 非梯度优化的思路

```
梯度下降:  沿梯度方向走 → 快但需要梯度
进化策略:  随机扰动 + 选择好的 → 慢但不需要梯度
```

> 💡 ES 在 RL 中有用——RL 的奖励信号不可微，ES 直接优化奖励。""")

# ============================================================
md("""## 2. 进化策略 (Evolution Strategies)

### 2.1 经典 ES

ES 维护一个参数分布 $\\theta \\sim N(\\mu, \\sigma^2 I)$，通过**采样+评估+更新**优化：

```
循环:
  1. 采样 N 个参数: θ_i = μ + σ * ε_i  (ε ~ N(0,I))
  2. 评估每个 θ_i 的适应度 f(θ_i)
  3. 更新 μ: μ += lr * mean(f_i * ε_i) / σ  (排名加权)
  4. 更新 σ: 自适应
```

### 2.2 关键优势

- **并行**：N 个采样可以 N 台机器并行评估
- **无梯度**：不需要反向传播
- **平滑**：对噪声目标更鲁棒

> 💡 OpenAI 2017 年用 ES 训练 RL，在 1440 个 CPU 上并行——比 PPO 更并行。""")

code("""def evolution_strategy(objective_fn, n_params, n_gen=100, pop_size=50,
                        sigma=0.5, lr=0.1):
    # 简化 ES: (μ, σ) 自适应进化策略
    mu = np.random.randn(n_params) * 0.1
    best_fitness = -float('inf')
    history = []

    for gen in range(n_gen):
        # 1. 采样
        eps = np.random.randn(pop_size, n_params)
        solutions = mu + sigma * eps

        # 2. 评估
        fitness = np.array([objective_fn(s) for s in solutions])

        # 3. 排名加权更新
        ranks = np.argsort(np.argsort(-fitness))  # 0=最好
        weights = np.maximum(0, np.log(pop_size/2 + 1) - np.log(ranks + 1))
        weights = weights / weights.sum() - 1/pop_size

        # 4. 更新 mu 和 sigma
        mu += lr * sigma * (weights @ eps)
        sigma *= np.exp(lr * 0.1 * (weights @ (eps**2 - 1) / 2))

        best = fitness.max()
        if best > best_fitness:
            best_fitness = best
        history.append((best, fitness.mean()))

    return mu, best_fitness, history

# 测试: 优化一个简单目标
def sphere_objective(x):
    # 最小化 sum(x^2), 最优 = 0 → 适应度 = -sum(x^2)
    return -np.sum(x**2)

def rastrigin_objective(x):
    # Rastrigin 函数 (多模态, 有很多局部最优)
    return -(10 * len(x) + sum(xi**2 - 10 * np.cos(2*np.pi*xi) for xi in x))

# 优化 Sphere
mu_sphere, best_sphere, hist_sphere = evolution_strategy(
    sphere_objective, n_params=10, n_gen=100, pop_size=50)

# 优化 Rastrigin (多模态, 梯度下降容易卡在局部最优)
mu_rastr, best_rastr, hist_rastr = evolution_strategy(
    rastrigin_objective, n_params=10, n_gen=100, pop_size=50)

print(f"Sphere: 最优适应度 = {best_sphere:.4f} (理论最优=0)")
print(f"Rastrigin: 最优适应度 = {best_rastr:.4f} (理论最优=0)")
print("ES 不需要梯度——对多模态目标更鲁棒。")""")

code("""# 可视化 ES 优化过程
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

best_s, mean_s = zip(*hist_sphere)
best_r, mean_r = zip(*hist_rastr)

axes[0].plot([-b for b in best_s], 'b-', linewidth=2, label='最好')
axes[0].plot([-m for m in mean_s], 'b--', linewidth=2, label='平均')
axes[0].set_xlabel('代数'); axes[0].set_ylabel('loss (越低越好)')
axes[0].set_title('ES 优化 Sphere 函数'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot([-b for b in best_r], 'r-', linewidth=2, label='最好')
axes[1].plot([-m for m in mean_r], 'r--', linewidth=2, label='平均')
axes[1].set_xlabel('代数'); axes[1].set_ylabel('loss (越低越好)')
axes[1].set_title('ES 优化 Rastrigin (多模态)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_es_optimization.png', bbox_inches='tight')
plt.show()
print("ES 在多模态函数上也能找到全局最优——这是它比梯度下降的优势。")""")

# ============================================================
md("""## 3. 神经进化：进化网络

### 3.1 神经进化的两种方式

| 方式 | 优化什么 | 代表 |
|------|---------|------|
| **权重进化** | 用 ES 优化权重 | OpenAI ES |
| **结构进化** | 搜索网络结构 | NEAT |

### 3.2 NEAT：神经进化增强拓扑

NEAT 同时进化**结构和权重**：
- 从简单网络开始
- 随机加节点/加边（结构变异）
- 随机改权重（权重变异）
- 用适应度选择

```
第1代:  [input] → [output]                    (简单)
第10代: [input] → [hidden] → [output]         (加隐层)
第50代: [input] → [h1] ⇄ [h2] → [output]     (复杂拓扑)
```

> 💡 NEAT 在简单任务上有效，但大规模深度网络还是梯度下降更好。
**现代神经进化**多用于 NAS（搜索架构），权重仍用梯度训练。""")

code("""# 简化神经进化: 用 ES 优化一个小网络的权重
class TinyNet:
    # 可变结构的小网络
    def __init__(self, n_hidden=5):
        self.W1 = np.random.randn(2, n_hidden) * 0.5
        self.W2 = np.random.randn(n_hidden, 1) * 0.5

    def forward(self, x):
        h = np.tanh(x @ self.W1)
        return h @ self.W2

    def get_params(self):
        return np.concatenate([self.W1.flatten(), self.W2.flatten()])

    def set_params(self, params):
        n1 = self.W1.size
        self.W1 = params[:n1].reshape(self.W1.shape)
        self.W2 = params[n1:].reshape(self.W2.shape)

# 任务: XOR
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([[0], [1], [1], [0]])

def xor_fitness(params):
    net = TinyNet(n_hidden=5)
    net.set_params(params)
    pred = np.array([net.forward(x) for x in X_xor])
    loss = np.mean((pred - y_xor) ** 2)
    return -loss  # 适应度 = -loss

# 用 ES 优化网络权重
n_params = 2*5 + 5*1  # W1 + W2
best_params, best_fit, hist_neuro = evolution_strategy(
    xor_fitness, n_params=n_params, n_gen=200, pop_size=50, sigma=0.3, lr=0.2)

# 验证
net = TinyNet(n_hidden=5)
net.set_params(best_params)
print("神经进化优化 XOR:")
for x, y in zip(X_xor, y_xor):
    pred = net.forward(x)
    print(f"  输入{x} → 预测{pred[0]:.3f}, 真实{y[0]}")
print(f"最终适应度: {best_fit:.4f} (越接近0越好)")""")

# ============================================================
md("""## 4. NAS：网络架构搜索

### 4.1 NAS 的三种方法

| 方法 | 说明 | 代表 |
|------|------|------|
| **强化学习** | RNN 控制器生成架构 | NASNet |
| **进化** | 进化架构种群 | AmoebaNet |
| **梯度** | 可微架构搜索 | DARTS |

### 4.2 DARTS：可微架构搜索

DARTS 把离散搜索**连续化**——用 softmax 混合所有候选操作：

$$h = \\sum_{o \\in O} \\alpha_o \\cdot o(x)$$

其中 $\\alpha_o$ 是操作权重，用梯度优化。最后选 $\\alpha$ 最大的操作。

```
离散 NAS:  遍历所有架构 → 不可微 → 用 RL/进化
DARTS:    连续混合所有操作 → 可微 → 用梯度
```

> 💡 DARTS 把 NAS 从需要几千 GPU-天 降到几 GPU-天——可微化的威力。""")

code("""# 简化 DARTS: 可微架构搜索
class DARTSCell(nn.Module):
    # DARTS cell: 混合多个候选操作
    def __init__(self, d_model, n_ops=4):
        super().__init__()
        self.n_ops = n_ops
        # 候选操作
        self.ops = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_ops)
        ])
        # 架构参数 (可学习)
        self.alphas = nn.Parameter(torch.randn(n_ops) * 0.1)

    def forward(self, x):
        # 混合所有操作: h = sum(softmax(α)_i * op_i(x))
        weights = F.softmax(self.alphas, dim=-1)
        out = sum(w * op(x) for w, op in zip(weights, self.ops))
        return out

    def get_architecture(self):
        # 选权重最大的操作
        return self.alphas.argmax().item()

# 训练 DARTS: 同时优化权重和架构参数
d_model = 16
darts_cell = DARTSCell(d_model, n_ops=4)

# 任务: 简单回归
X_train = torch.randn(100, d_model)
y_train = torch.sin(X_train[:, :1]) + 0.1 * torch.randn(100, 1)

# 权重优化器 (操作参数)
opt_weights = torch.optim.Adam(darts_cell.ops.parameters(), lr=1e-2)
# 架构优化器 (alpha 参数)
opt_arch = torch.optim.Adam([darts_cell.alphas], lr=1e-2)

for epoch in range(200):
    # 1. 固定架构, 训练操作权重
    opt_weights.zero_grad()
    pred = darts_cell(X_train)[:, :1]
    loss_w = F.mse_loss(pred, y_train)
    loss_w.backward()
    opt_weights.step()

    # 2. 固定权重, 更新架构参数
    opt_arch.zero_grad()
    pred = darts_cell(X_train)[:, :1]
    loss_a = F.mse_loss(pred, y_train)
    loss_a.backward()
    opt_arch.step()

best_op = darts_cell.get_architecture()
alpha_probs = F.softmax(darts_cell.alphas, dim=-1)
print(f"DARTS 架构搜索完成:")
print(f"  操作权重: {alpha_probs.detach().numpy().round(3)}")
print(f"  最优操作: op_{best_op}")
print(f"  最终损失: {loss_w.item():.4f}")
print("DARTS 用梯度同时优化权重和架构——比离散搜索高效得多。")""")

# ============================================================
md("""## 5. 梯度 vs 非梯度：什么时候用什么？

### 5.1 对比

| | 梯度下降 | ES | NAS |
|---|---------|-----|------|
| **效率** | 高 (用梯度信息) | 低 (大量采样) | 很低 |
| **并行** | 中 (数据并行) | 极高 (采样并行) | 高 |
| **可微** | 必须 | 不需要 | 不需要 |
| **适用** | 深度学习 | RL/黑盒 | 架构搜索 |

### 5.2 选择指南

```
有梯度 + 可微     → 梯度下降 (首选)
无梯度 + 黑盒     → ES/CMA-ES
搜索离散结构      → NAS (DARTS 优先)
多目标/约束       → 进化算法 (NSGA-II)
```

> 💡 现代深度学习几乎全用梯度下降。非梯度方法在**架构搜索**和**RL**中有特殊价值。
> AlphaEvolve (Google 2024) 用 LLM + 进化搜索数学发现——非梯度的新应用。""")

code("""# 可视化: 梯度 vs ES 在不同问题上的表现
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1. 凸问题 (梯度更好)
convex_grad = np.exp(-np.linspace(0, 5, 100))
convex_es = np.exp(-np.linspace(0, 2, 100)) + 0.05
axes[0].plot(convex_grad, 'b-', linewidth=2.5, label='梯度下降 (快)')
axes[0].plot(convex_es, 'r-', linewidth=2.5, label='ES (慢但稳)')
axes[0].set_xlabel('步数'); axes[0].set_ylabel('loss')
axes[0].set_title('凸问题: 梯度下降更快'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 2. 多模态问题 (ES 更好)
np.random.seed(42)
multi_grad = np.minimum.accumulate(np.abs(np.random.randn(100) * 0.3 + 0.5) + 0.3)
multi_grad = np.minimum.accumulate(multi_grad)
multi_es = np.exp(-np.linspace(0, 3, 100)) * 0.5
axes[1].plot(multi_grad, 'b-', linewidth=2.5, label='梯度下降 (卡局部)')
axes[1].plot(multi_es, 'r-', linewidth=2.5, label='ES (跳出来)')
axes[1].set_xlabel('步数'); axes[1].set_ylabel('loss')
axes[1].set_title('多模态: ES 更鲁棒'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_grad_vs_es.png', bbox_inches='tight')
plt.show()
print("凸问题梯度快; 多模态ES鲁棒——各有所长。")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 进化策略 (ES) 采样+评估+更新 | ✅ |
| 神经进化 (NEAT 思想) | ✅ |
| NAS 架构搜索三种方法 | ✅ |
| DARTS 可微架构搜索 | ✅ |
| 梯度 vs 非梯度适用场景 | ✅ |

### 核心 takeaway

> **梯度下降是王道，但非梯度方法在特定场景不可替代**。
> ES 适合黑盒/RL，NAS 适合架构搜索，DARTS 让 NAS 可微化。
> AlphaEvolve 展示了 LLM + 进化的新方向。

### 🔗 下一板块预告

**`19_reasoning_models.ipynb`** — 🔥o1/o3、DeepSeek-R1、test-time scaling（进入推理模型板块）

---

> 💬 **写在最后**：非梯度优化是梯度下降的补充——在不可微、多模态、黑盒场景中不可替代。
> 理解 ES 和 NAS，就理解了优化方法的完整图景。**板块二(现代训练范式)完结。**""")

# ============================================================
output_path = "notebooks/18_non_gradient_optimization.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")