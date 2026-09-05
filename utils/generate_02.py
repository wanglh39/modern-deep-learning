"""生成 02_autograd_optimizers.ipynb 的脚本"""
import json
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
# 0. 概述
# ============================================================
md("""# 02 — 自动微分引擎、优化器演进与正则化

> 上一章我们**手写**了反向传播。每写一层都要手动算 $\\delta$、手动传梯度。
> 你肯定在想：能不能让计算机自动做这件事？——这就是**自动微分（autograd）**。
>
> 本章我们：1) 自己造一个迷你 autograd 引擎，理解 PyTorch 的 `backward()` 背后发生了什么；
> 2) 从 SGD 一路演进到 AdamW，看清楚每个优化器解决了前一个的什么痛点；
> 3) 学会 Dropout、Weight Decay 等正则化手段。

## 本章你将掌握

1. **自动微分引擎原理**：计算图自动构建 + 反向模式自动微分
2. **从零实现一个迷你 autograd**（Tensor 类，支持 backward）
3. **优化器演进**：SGD → Momentum → RMSProp → Adam → AdamW，每个为什么比上一个强
4. **正则化**：Dropout、Weight Decay（L2）、Early Stopping
5. 在同一问题上**可视化对比**不同优化器的收敛轨迹""")

# ============================================================
# 1. 环境准备
# ============================================================
md("## 0. 环境准备")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
print("环境准备完成 ✅")""")

# ============================================================
# 2. 自动微分原理
# ============================================================
md("""## 1. 自动微分引擎原理

### 1.1 上一章的痛点

上一章我们手写反向传播时，每加一层都要：
1. 手动算 $\\delta = (W^\\top \\delta_{\\text{上游}}) \\odot f'(z)$
2. 手动算 $\\frac{\\partial L}{\\partial W} = \\delta a^\\top$

这很机械、很容易写错。**自动微分**的核心思想是：

> 每次做前向运算时，**顺便记录下这个运算的局部导数**，存进计算图。
> 反向传播时，沿着图反向走，每个节点只需用自己的局部导数乘以上游梯度。

### 1.2 两种自动微分模式

| 模式 | 方向 | 特点 |
|------|------|------|
| **前向模式** | 输入 → 输出 | 对每个输入单独算 $\\frac{\\partial y}{\\partial x_i}$，适合输入少输出多 |
| **反向模式** | 输出 → 输入 | 一次反向就算出所有 $\\frac{\\partial L}{\\partial \\theta_i}$，适合输入多输出少 |

深度学习参数多、损失只有一个标量 → **反向模式**完胜。PyTorch 的 `backward()` 就是反向模式。

### 1.3 反向模式的核心：每个节点存一个 `grad_fn`

```
c = a + b   →  c.grad_fn = "加法"，c 的上游梯度均分给 a 和 b
d = a * b   →  d.grad_fn = "乘法"，d 的上游梯度乘以 b 给 a，乘以 a 给 b
```

反向传播时，从输出开始，每个节点调用自己的 `grad_fn` 把梯度传给输入。""")

# ============================================================
# 3. 从零实现迷你 autograd
# ============================================================
md("""## 2. 从零实现一个迷你 autograd 引擎

我们实现一个 `Tensor` 类，支持：
- 基本运算（加、乘、矩阵乘、ReLU）
- 自动构建计算图
- `backward()` 自动算所有梯度

**这是 PyTorch `torch.Tensor` 的灵魂简化版**。""")

code("""class Tensor:
    \"\"\"
    迷你 autograd Tensor —— PyTorch 的灵魂简化版

    每个 Tensor 记录：
    - data: 数值
    - grad: 梯度（反向传播后填充）
    - _backward: 这个节点的反向传播函数
    - _prev: 父节点（计算图中的输入）
    - requires_grad: 是否需要算梯度
    \"\"\"
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None   # 默认空操作
        self._prev = set(_children)      # 计算图中的父节点
        self._op = _op                   # 产生这个节点的运算（调试用）

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    # ---------- 加法 ----------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward
        return out

    # ---------- 乘法（逐元素）----------
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # ---------- 矩阵乘 ----------
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='@')

        def _backward():
            # 矩阵乘的链式法则
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # ---------- ReLU ----------
    def relu(self):
        out = Tensor(np.maximum(0, self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='relu')

        def _backward():
            # ReLU 的导数：正数传梯度，负数挡住
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    # ---------- 平方和（常用损失）----------
    def sum_of_squares(self):
        out = Tensor(np.sum(self.data ** 2),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='sum_sq')

        def _backward():
            self.grad += 2 * self.data * out.grad
        out._backward = _backward
        return out

    # ---------- 反向传播 ----------
    def backward(self):
        \"\"\"反向模式自动微分：拓扑排序后逆序执行每个节点的 _backward\"\"\"
        # 1. 拓扑排序（把计算图排成线性顺序）
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 2. 输出节点的梯度 = 1
        self.grad = np.ones_like(self.data)

        # 3. 逆序执行每个节点的反向传播
        for v in reversed(topo):
            v._backward()

    # 便捷方法
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __pow__(self, n): return Tensor(self.data ** n, _children=(self,), _op=f'**{n}')

print("Tensor 类定义完成 ✅")
print("这就是 PyTorch autograd 的核心思想——每个运算记录局部导数，反向时自动链式传播。")""")

md("""### 2.1 用迷你 autograd 算梯度

我们来算一个具体的例子，验证引擎工作正常。

设 $f(x, y) = (x + y)^2 + 3xy$，在 $x=2, y=3$ 处求梯度。

手算：$\\frac{\\partial f}{\\partial x} = 2(x+y) + 3y = 2(5) + 9 = 19$，$\\frac{\\partial f}{\\partial y} = 2(x+y) + 3x = 10 + 6 = 16$""")

code("""# 用迷你 autograd 算梯度
x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)

# f = (x + y)^2 + 3*x*y
s = x + y           # s = x + y
f = s * s + (x * y * 3)  # f = s² + 3xy

print(f"f = {f.data}")
f.backward()
print(f"df/dx = {x.grad}  (手算: 19)")
print(f"df/dy = {y.grad}  (手算: 16)")
print(f"✅ 自动微分结果与手算一致！")""")

md("""### 2.2 用迷你 autograd 训练一个 MLP

既然有了 autograd，我们就不用手写反向传播了——定义前向传播，loss.backward()，梯度自动算好。""")

code("""# 用迷你 autograd 实现一个 MLP 层
class Linear:
    \"\"\"线性层，参数是 Tensor\"\"\"
    def __init__(self, in_dim, out_dim):
        # He 初始化
        scale = np.sqrt(2.0 / in_dim)
        self.W = Tensor(np.random.randn(out_dim, in_dim) * scale, requires_grad=True)
        self.b = Tensor(np.zeros((out_dim, 1)), requires_grad=True)

    def __call__(self, x):
        # x: Tensor, 形状 (in_dim, 1)
        return (self.W @ x + self.b).relu()  # 简化：含 ReLU

    def params(self):
        return [self.W, self.b]

# 构建网络: 2 → 8 → 1
np.random.seed(42)
layer1 = Linear(2, 8)
layer2 = Linear(8, 1)

# 一个训练样本
x_data = np.array([[0.5], [-0.3]])
y_data = np.array([[1.0]])

# 前向传播（全程自动构建计算图）
x = Tensor(x_data, requires_grad=False)
h = layer1(x)          # 隐藏层
y_hat = layer2(h)      # 输出层（含 relu，简化版）

# 损失 = 0.5 * (y_hat - y)²
loss = ((y_hat - Tensor(y_data)) * (y_hat - Tensor(y_data))).sum_of_squares() * 0.5
# 修正：用更简单的方式
diff = y_hat - Tensor(y_data)
loss = diff * diff

# 反向传播（一行！）
loss.backward()

print(f"loss = {loss.data}")
print(f"layer1.W 的梯度形状: {layer1.W.grad.shape}")
print(f"layer2.W 的梯度形状: {layer2.W.grad.shape}")
print("✅ autograd 自动算出了所有参数的梯度——不用手写反向传播了！")""")

# ============================================================
# 4. PyTorch autograd 对比
# ============================================================
md("""## 3. PyTorch 的 autograd：工业级实现

我们的迷你 Tensor 只有 100 行代码，PyTorch 的实现有数万行，但**核心思想完全一样**：

1. 前向运算时记录 `grad_fn`（局部导数）
2. `backward()` 时拓扑排序 + 逆序传播

来看 PyTorch 的版本，体会一下""")

code("""import torch

# 同样的函数 f = (x + y)² + 3xy
x = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
y = torch.tensor(3.0, requires_grad=True, dtype=torch.float64)

s = x + y
f = s * s + x * y * 3

print(f"PyTorch 计算的 f = {f.item()}")
f.backward()
print(f"df/dx = {x.grad.item()}  (手算: 19)")
print(f"df/dy = {y.grad.item()}  (手算: 16)")

# 看看 PyTorch 记算图里的 grad_fn
print(f"\\nPyTorch 计算图节点:")
print(f"  s = x + y  →  s.grad_fn = {s.grad_fn}")
print(f"  f = s*s + ...  →  f.grad_fn = {f.grad_fn}")
print("每个 grad_fn 就是我们迷你版里的 _backward！")""")

# ============================================================
# 5. 优化器演进
# ============================================================
md("""## 4. 优化器演进：SGD → Momentum → RMSProp → Adam → AdamW

这是本章的重头戏。我们用一个**统一的可视化**来对比每个优化器。

### 4.1 先理解问题：梯度下降在做什么

$$\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)$$

就是沿着梯度的**反方向**走一步。但这个简单公式有很多问题，优化器的演进就是在解决这些问题。""")

code("""# 准备一个测试函数：Rosenbrock 函数（经典的优化测试函数）
# f(x, y) = (1 - x)² + 100(y - x²)²
# 最小值在 (1, 1)，但有个弯曲的谷，梯度下降很难走

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dfdx = -2*(1 - x) - 400*x*(y - x**2)
    dfdy = 200*(y - x**2)
    return np.array([dfdx, dfdy])

# 可视化 Rosenbrock 函数的等高线
x_range = np.linspace(-1.5, 2.0, 100)
y_range = np.linspace(-0.5, 2.5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.5)
ax.plot(1, 1, 'r*', markersize=15, label='最小值 (1,1)')
ax.set_title("Rosenbrock 函数（弯曲的谷，优化器噩梦）")
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend()
plt.tight_layout()
plt.show()
print("Rosenbrock 函数：最小值在 (1,1)，但梯度几乎垂直于谷底方向，普通 SGD 会剧烈震荡。")""")

md("""### 4.2 SGD（随机梯度下降）

$$\\theta_{t+1} = \\theta_t - \\eta \\nabla L$$

**痛点**：在 Rosenbrock 这种弯曲谷里，梯度方向几乎垂直于谷底，SGD 会剧烈震荡，沿着谷的方向却走得很慢。""")

code("""def optimize(optimizer_name, init_pos, n_steps, lr=0.001):
    \"\"\"用指定优化器优化 Rosenbrock，记录轨迹\"\"\"
    pos = init_pos.copy()
    trajectory = [pos.copy()]

    # 各优化器的状态
    velocity = np.zeros(2)       # Momentum 用
    sq_grad = np.zeros(2)        # RMSProp/Adam 用
    m = np.zeros(2)              # Adam 的一阶矩
    v = np.zeros(2)              # Adam 的二阶矩
    beta1, beta2 = 0.9, 0.999    # Adam 的超参

    for t in range(1, n_steps + 1):
        grad = rosenbrock_grad(pos[0], pos[1])

        if optimizer_name == 'SGD':
            pos -= lr * grad

        elif optimizer_name == 'Momentum':
            velocity = 0.9 * velocity + grad
            pos -= lr * velocity

        elif optimizer_name == 'RMSProp':
            sq_grad = 0.9 * sq_grad + 0.1 * grad**2
            pos -= lr * grad / (np.sqrt(sq_grad) + 1e-8)

        elif optimizer_name == 'Adam':
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)   # 偏差修正
            v_hat = v / (1 - beta2**t)
            pos -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        elif optimizer_name == 'AdamW':
            # AdamW = Adam + 解耦的 weight decay
            weight_decay = 0.01
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            pos -= lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + weight_decay * pos)

        trajectory.append(pos.copy())

    return np.array(trajectory)

# 在同一起点对比所有优化器
init = np.array([-1.0, 1.0])
lr = 0.002
steps = 300

fig, ax = plt.subplots(figsize=(10, 7))
ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.3)
ax.plot(1, 1, 'r*', markersize=15, zorder=5)

colors = {'SGD': '#E74C3C', 'Momentum': '#3498DB', 'RMSProp': '#2ECC71',
          'Adam': '#9B59B6', 'AdamW': '#F39C12'}

for name in ['SGD', 'Momentum', 'RMSProp', 'Adam', 'AdamW']:
    traj = optimize(name, init, steps, lr)
    ax.plot(traj[:, 0], traj[:, 1], color=colors[name], linewidth=2, label=name, alpha=0.8)
    ax.plot(traj[0, 0], traj[0, 1], 'o', color=colors[name], markersize=8)

ax.set_title("优化器对比：在 Rosenbrock 函数上的收敛轨迹")
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('notebooks/fig_optimizer_comparison.png', bbox_inches='tight')
plt.show()
print("观察：SGD 震荡最严重，Adam/AdamW 收敛最快最稳。")""")

md("""### 4.3 每个优化器解决了什么问题

| 优化器 | 公式 | 解决了什么痛点 |
|--------|------|---------------|
| **SGD** | $\\theta \\leftarrow \\theta - \\eta \\nabla L$ | 基础版，但在谷里震荡 |
| **Momentum** | $v \\leftarrow \\beta v + \\nabla L$;<br>$\\theta \\leftarrow \\theta - \\eta v$ | 累积历史梯度，**抑制震荡、加速沿谷方向** |
| **RMSProp** | $s \\leftarrow \\beta s + (1-\\beta)\\nabla L^2$;<br>$\\theta \\leftarrow \\theta - \\eta \\nabla L / \\sqrt{s}$ | **自适应学习率**：梯度大的维度步长缩小 |
| **Adam** | 结合 Momentum + RMSProp | 既有动量加速，又有自适应学习率 |
| **AdamW** | Adam + **解耦的 weight decay** | 修正 Adam + L2 正则化的理论缺陷 |

> 💡 **Adam 是当前深度学习最常用的优化器**。AdamW 在大模型训练中更受青睐，
> 因为它修正了 weight decay 与自适应学习率耦合的问题。""")

code("""# 量化对比：每个优化器到达 (1,1) 的速度
fig, ax = plt.subplots(figsize=(10, 5))

for name in ['SGD', 'Momentum', 'RMSProp', 'Adam', 'AdamW']:
    traj = optimize(name, init, steps, lr)
    dist = np.sqrt((traj[:, 0] - 1)**2 + (traj[:, 1] - 1)**2)
    ax.plot(dist, color=colors[name], linewidth=2, label=name)

ax.set_title('到最小值的距离 vs 步数')
ax.set_xlabel('步数'); ax.set_ylabel('距离')
ax.set_yscale('log')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_optimizer_convergence.png', bbox_inches='tight')
plt.show()
print("Adam/AdamW 收敛最快——这就是为什么它们是默认选择。")""")

# ============================================================
# 6. 正则化
# ============================================================
md("""## 5. 正则化：让模型别过拟合

### 5.1 过拟合是什么

模型把训练数据**背下来**了，但没学到规律，在新数据上表现差。

```
欠拟合 ←—————→ 刚好 ←—————→ 过拟合
  太简单           ✓           把噪声也学了
```

### 5.2 三种正则化手段

| 手段 | 原理 | 实现 |
|------|------|------|
| **Weight Decay (L2)** | 惩罚大权重：$L + \\lambda \\|w\\|^2$ | 优化器里加 `weight_decay` |
| **Dropout** | 训练时随机置零一部分神经元 | `nn.Dropout(p)` |
| **Early Stopping** | 验证集 loss 不再下降就停 | 监控 val_loss |""")

code("""import torch
import torch.nn as nn
import torch.optim as optim

# 生成一个容易过拟合的数据集（少量样本 + 噪声）
np.random.seed(42)
n_train, n_val = 20, 50
x_all = np.linspace(-3, 3, n_train + n_val)
y_all = np.sin(x_all) + np.random.randn(len(x_all)) * 0.3

x_train = torch.tensor(x_all[:n_train], dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(y_all[:n_train], dtype=torch.float32).reshape(-1, 1)
x_val = torch.tensor(x_all[n_train:], dtype=torch.float32).reshape(-1, 1)
y_val = torch.tensor(y_all[n_train:], dtype=torch.float32).reshape(-1, 1)

def train_model(use_dropout=False, use_weight_decay=False, epochs=500):
    \"\"\"训练模型，返回训练/验证损失曲线\"\"\"
    torch.manual_seed(0)
    layers = [nn.Linear(1, 64), nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    layers += [nn.Linear(64, 64), nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.Linear(64, 1))
    model = nn.Sequential(*layers)

    wd = 0.01 if use_weight_decay else 0.0
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=wd)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(x_val), y_val).item())

    return train_losses, val_losses, model

# 对比：无正则化 vs Dropout vs Weight Decay
configs = [
    ('无正则化', False, False),
    ('Dropout', True, False),
    ('Weight Decay', False, True),
    ('Dropout + WD', True, True),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_cfg = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6']

for (name, do_drop, do_wd), color in zip(configs, colors_cfg):
    tl, vl, model = train_model(do_drop, do_wd)
    axes[0].plot(tl, color=color, linewidth=2, label=name)
    axes[1].plot(vl, color=color, linewidth=2, label=name)

axes[0].set_title('训练损失'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[1].set_title('验证损失（关键看这个）'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
for ax in axes:
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.5)
plt.tight_layout()
plt.savefig('notebooks/fig_regularization_comparison.png', bbox_inches='tight')
plt.show()
print("观察：无正则化的训练 loss 最低，但验证 loss 最高 → 过拟合！正则化缓解了这个问题。")""")

md("""### 5.3 Dropout 的直觉

Dropout 像是在每次训练时**随机删掉一部分神经元**，强迫网络不要依赖任何单个神经元。

> 想象一个团队，如果每次随机有人请假，其他人就得学会**谁的工作都能做**。
> 这样团队就不依赖任何单个人——这就是 Dropout 防止过拟合的直觉。""")

code("""# 可视化 Dropout 的效果
torch.manual_seed(42)
dropout = nn.Dropout(0.5)
x_demo = torch.ones(10)

dropout.train()  # 训练模式：会 dropout
print("训练模式（随机置零）:")
for i in range(3):
    out = dropout(x_demo.clone())
    print(f"  第 {i+1} 次: {out.numpy().astype(int)}")

dropout.eval()   # 评估模式：不 dropout
print(f"\\n评估模式（恒等变换）: {dropout(x_demo.clone()).numpy()}")""")

# ============================================================
# 7. 综合实验
# ============================================================
md("""## 6. 综合实验：在月牙数据上对比

把优化器和正则化都用上，在一个真实分类任务上对比。""")

code("""# 月牙数据
def make_moons(n=200, noise=0.2):
    t = np.linspace(0, np.pi, n//2)
    X = np.vstack([
        np.column_stack([np.cos(t), np.sin(t)]),
        np.column_stack([1-np.cos(t), -np.sin(t)])
    ])
    X += np.random.randn(n, 2) * noise
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

X, y = make_moons(300)
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

def train_and_eval(optimizer_name, epochs=200):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(32, 1))

    opts = {
        'SGD': optim.SGD(model.parameters(), lr=0.1),
        'Momentum': optim.SGD(model.parameters(), lr=0.1, momentum=0.9),
        'RMSProp': optim.RMSprop(model.parameters(), lr=0.01),
        'Adam': optim.Adam(model.parameters(), lr=0.01),
        'AdamW': optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01),
    }
    optimizer = opts[optimizer_name]
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses, model

fig, ax = plt.subplots(figsize=(10, 5))
for name in ['SGD', 'Momentum', 'RMSProp', 'Adam', 'AdamW']:
    losses, _ = train_and_eval(name)
    ax.plot(losses, color=colors[name], linewidth=2, label=name)
ax.set_title('月牙分类：不同优化器的训练损失')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_optimizer_moons.png', bbox_inches='tight')
plt.show()
print("在真实任务上，Adam 和 AdamW 依然表现最好。")""")

# ============================================================
# 8. 小结
# ============================================================
md("""## 7. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 自动微分原理（反向模式） | ✅ |
| 从零实现迷你 autograd 引擎（Tensor 类） | ✅ |
| PyTorch autograd 对比验证 | ✅ |
| SGD → Momentum → RMSProp → Adam → AdamW 演进 | ✅ |
| 每个优化器解决了什么痛点 | ✅ |
| 优化器收敛轨迹可视化对比 | ✅ |
| Weight Decay、Dropout、Early Stopping 正则化 | ✅ |
| 过拟合与正则化的关系 | ✅ |

### 🔗 下一章预告

**`03_normalization_activation.ipynb`** — 归一化(BN→LN→RMSNorm) + 激活函数(ReLU→GELU→SwiGLU) + 初始化

我们将：
- 理解为什么需要归一化（内部协变量偏移）
- BatchNorm → LayerNorm → RMSNorm 的演进（为什么 LLM 用 RMSNorm）
- ReLU → GELU → SwiGLU 的演进（为什么 LLM 用 SwiGLU）
- 参数初始化为什么重要（Xavier vs He 初始化）

### 📚 延伸阅读

- **PyTorch autograd 文档**：官方实现细节
- **Adam 原论文**（Kingma & Ba, 2014）：自适应矩估计
- **AdamW 论文**（Loshchilov & Hutter, 2019）：解耦的 weight decay

---

> 💬 **写在最后**：你刚从零实现了一个 autograd 引擎——这就是 PyTorch 最核心的机制。
> 以后写 `loss.backward()` 时，你知道里面在做拓扑排序 + 逆序传播每个节点的局部导数。
> 优化器方面，记住一句话：**Adam 是默认选择，AdamW 是大模型训练的选择**。""")

# ============================================================
# 写入文件
# ============================================================
output_path = "notebooks/02_autograd_optimizers.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")