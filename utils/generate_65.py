# 生成 65_molecule_pinn.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 65 — 分子生成与 PINN

> 用神经网络解微分方程、生成分子——AI 正在改变计算科学。

## 本章你将掌握

1. **PINN**：物理信息神经网络
2. **神经算子**：学习算子映射
3. **分子生成**：用 AI 设计新分子
4. **应用**：药物发现、材料模拟""")

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

md("""## 1. PINN：物理信息神经网络

### 1.1 核心思想

```
PINN (Physics-Informed Neural Network):
  用神经网络解偏微分方程 (PDE)

传统方法:
  有限元/有限差分 → 离散网格 → 解
  问题: 网格生成复杂、高维困难

PINN:
  u(x, t) = NN(x, t)  # 神经网络近似解
  约束: PDE(NN) = 0    # PDE 作为损失

损失 = 数据损失 + PDE 损失 + 边界损失
```

### 1.2 示例：热方程

```
热方程: ∂u/∂t = α ∂²u/∂x²

PINN:
  u = NN(x, t)
  PDE 损失: ∂u/∂t - α ∂²u/∂x² = 0
  
  用自动微分计算导数!
```""")

code("""# PINN 解热方程
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

# 热方程: u_t = alpha * u_xx
alpha = 0.1
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练
for epoch in range(1000):
    # 随机采样
    x = torch.rand(100, 1, requires_grad=True)
    t = torch.rand(100, 1, requires_grad=True)

    u = model(x, t)

    # 自动微分
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # PDE 损失
    pde_loss = F.mse_loss(u_t, alpha * u_xx)

    # 初始条件: u(x, 0) = sin(pi*x)
    x_ic = torch.rand(50, 1)
    t_ic = torch.zeros(50, 1)
    u_ic = model(x_ic, t_ic)
    ic_loss = F.mse_loss(u_ic, torch.sin(np.pi * x_ic))

    # 边界条件: u(0, t) = u(1, t) = 0
    t_bc = torch.rand(20, 1)
    u_bc1 = model(torch.zeros(20, 1), t_bc)
    u_bc2 = model(torch.ones(20, 1), t_bc)
    bc_loss = F.mse_loss(u_bc1, torch.zeros_like(u_bc1)) + F.mse_loss(u_bc2, torch.zeros_like(u_bc2))

    loss = pde_loss + ic_loss + bc_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"PINN 训练完成, 最终损失: {loss.item():.6f}")

# 可视化
x_plot = torch.linspace(0, 1, 50).unsqueeze(1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, t_val in zip(axes, [0.0, 0.2, 0.5]):
    t_plot = torch.full((50, 1), t_val)
    with torch.no_grad():
        u_pred = model(x_plot, t_plot).numpy()
    u_exact = np.sin(np.pi * x_plot.numpy().flatten()) * np.exp(-alpha * np.pi**2 * t_val)
    ax.plot(x_plot.numpy(), u_pred, 'b-', linewidth=2, label='PINN')
    ax.plot(x_plot.numpy(), u_exact, 'r--', linewidth=2, label='解析解')
    ax.set_title(f't={t_val}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlabel('x'); ax.set_ylabel('u')

plt.suptitle('PINN 解热方程', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/fig_pinn.png', bbox_inches='tight')
plt.show()""")

md("""## 2. 神经算子

### 2.1 从函数到算子

```
PINN: 学习一个函数 u(x)
神经算子: 学习一个算子 G: a(x) → u(x)

  输入: 函数 a(x) (如初始条件)
  输出: 函数 u(x) (如解)

  → 一次训练，任意输入

代表: DeepONet, Fourier Neural Operator (FNO)

优势:
  - 不需要重新训练 (零样本)
  - 分辨率无关
  - 高维问题
```

### 2.2 FNO

```
FNO (Fourier Neural Operator):
  在频域做卷积 → 高效

  1. 输入 → FFT (傅里叶变换)
  2. 频域线性变换
  3. → IFFT (逆变换)
  4. + 非线性激活

  → 全局信息 (频域) + 局部信息 (残差)
```""")

code("""# FNO 简化实现
class FNO(nn.Module):
    def __init__(self, modes=8, width=32):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(2, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

        # 频域权重
        self.weights = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) * 0.01)

    def forward(self, x):
        # x: (B, N, 2) 位置+输入
        B, N, _ = x.shape

        # 提升维度
        x = self.fc0(x)  # (B, N, width)

        # FFT
        x_ft = torch.fft.fft(x, dim=1)  # (B, N, width) complex

        # 频域变换 (只保留低频 modes)
        x_ft_out = torch.zeros_like(x_ft)
        for i in range(self.modes):
            x_ft_out[:, i, :] = torch.matmul(x_ft[:, i, :], self.weights[:, :, i])

        # IFFT
        x = torch.fft.ifft(x_ft_out, dim=1).real

        # 降维
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 演示
fno = FNO(modes=8, width=16)
x_input = torch.randn(10, 32, 2)
with torch.no_grad():
    output = fno(x_input)
print(f"FNO: 输入 {x_input.shape} → 输出 {output.shape}")
print("FNO 在频域做卷积，高效处理全局信息。")""")

md("""## 3. 分子生成

### 3.1 用 AI 设计分子

```
分子生成:
  目标: 生成有特定性质的分子
  - 药物: 结合特定靶点
  - 材料: 有特定性质
  - 催化剂: 高活性

方法:
  1. SMILES 生成 (文本)
  2. 图生成 (分子图)
  3. 3D 生成 (空间结构)
```""")

code("""# 分子生成 (SMILES 简化)
class MoleculeGenerator(nn.Module):
    def __init__(self, vocab_size=30, d_model=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.rnn(emb)
        return self.head(out)

    def generate(self, start_token, max_len=20):
        tokens = [start_token]
        for _ in range(max_len):
            x = torch.tensor([tokens])
            with torch.no_grad():
                logits = self.forward(x)
            next_token = logits[0, -1].argmax().item()
            tokens.append(next_token)
            if next_token == 0:  # 结束符
                break
        return tokens

# 模拟 SMILES 字符表
smiles_chars = list('CCNOC(=O)NH12345')
gen = MoleculeGenerator(vocab_size=len(smiles_chars))

# 生成分子
print("分子生成:")
for i in range(3):
    start = np.random.randint(0, len(smiles_chars))
    tokens = gen.generate(start, max_len=10)
    smiles = ''.join([smiles_chars[t] for t in tokens if t < len(smiles_chars)])
    print(f"  分子 {i+1}: {smiles}")""")

md("""## 4. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| PINN (物理信息神经网络) | ✅ |
| 神经算子 (FNO/DeepONet) | ✅ |
| 分子生成 | ✅ |

### 核心 takeaway
> **PINN 把 PDE 编码进损失，神经算子学习算子映射**——AI 正在改变计算科学，从解方程到设计分子。

### 🔗 下一章
**`66_materials_discovery.ipynb`** — GNoME 材料发现

---

> 💬 **板块十一(AI for Science)进行中 (3/6)。**""")

output_path = "notebooks/65_molecule_pinn.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")