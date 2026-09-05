# 生成 69_sae_interpretability.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 69 — 稀疏自编码器与机制可解释性

> 🔥 打开黑盒：用稀疏自编码器发现 LLM 内部的"概念"。

## 本章你将掌握

1. **机制可解释性**：不只是"什么重要"，而是"为什么"
2. **稀疏自编码器 (SAE)**：发现可解释特征
3. **特征可视化**：看模型内部的"想法"
4. **应用**：监控、调试、安全""")

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

md("""## 1. 机制可解释性

### 1.1 从黑盒到白盒

```
传统可解释性 (注意:
  "哪些输入特征重要?"
  → 梯度、注意力权重

机制可解释性 (Mechanistic):
  "模型内部如何计算?"
  → 逆向工程，找计算电路

目标:
  把神经网络分解为可理解的"电路"
  - 每个电路实现一个功能
  - 电路间如何交互
  → 理解模型"为什么"这么算
```

### 1.2 核心问题

```
1. 特征: 神经元/激活代表什么概念?
2. 电路: 哪些神经元组合实现什么功能?
3. 计算: 信息如何流过网络?
4. 行为: 为什么模型对某输入有某输出?

→ 把"黑盒"变成"透明盒"
```

> 💡 机制可解释性是 AI 安全的关键——理解模型才能信任模型。""")

md("""## 2. 稀疏自编码器 (SAE)

### 2.1 为什么需要 SAE？

```
问题: 神经元是"多义的" (Polysemantic)
  一个神经元可能对多个概念都有反应
  → 难以解释

SAE 解决方案:
  把激活分解为"单义"特征
  → 每个特征只代表一个概念

  激活 x ≈ Σ f_i * d_i
  f_i: 特征激活 (稀疏，大部分为0)
  d_i: 特征方向 (可解释)
```

### 2.2 SAE 架构

```
SAE (Sparse Autoencoder):
  编码: f = ReLU(W_enc * x + b_enc)
  解码: x_hat = W_dec * f + b_dec

  损失 = ||x - x_hat||² + λ * ||f||_1
         ^^^^^^^^^^^^^^^^    ^^^^^^^^
         重建损失             稀疏惩罚

  → f 是稀疏的 (大部分特征不激活)
  → 每个激活的特征有明确含义
```""")

code("""# SAE 实现
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, n_features, sparsity_lambda=0.01):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim)
        self.lam = sparsity_lambda

    def forward(self, x):
        # 编码
        f = F.relu(self.encoder(x))  # 稀疏特征
        # 解码
        x_hat = self.decoder(f)  # 重建
        return f, x_hat

    def loss(self, x):
        f, x_hat = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = f.mean()  # L1 稀疏
        return recon_loss + self.lam * sparsity_loss, recon_loss, sparsity_loss

# 训练 SAE
input_dim = 32
n_features = 128  # 过完备 (特征 > 输入维度)

sae = SparseAutoencoder(input_dim, n_features, sparsity_lambda=0.1)
optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

# 模拟 LLM 激活
np.random.seed(42)
activations = torch.randn(1000, input_dim) * 2

for epoch in range(500):
    total_loss, recon, sparse = sae.loss(activations)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

print(f"SAE 训练完成:")
print(f"  输入维度: {input_dim}")
print(f"  特征数: {n_features} (过完备)")
print(f"  重建损失: {recon.item():.6f}")
print(f"  稀疏损失: {sparse.item():.6f}")

# 分析特征
with torch.no_grad():
    features, _ = sae(activations)

# 每个特征的激活频率
activation_freq = (features > 0).float().mean(dim=0)
active_features = (activation_freq > 0.01).sum().item()

print(f"\\n特征分析:")
print(f"  活跃特征数: {active_features}/{n_features}")
print(f"  平均稀疏度: {(features == 0).float().mean():.2%}")""")

code("""# 可视化 SAE 特征
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 特征激活频率
ax = axes[0]
freq = activation_freq.numpy()
ax.bar(range(n_features), freq, color='steelblue', alpha=0.8)
ax.set_xlabel('特征索引', fontsize=12)
ax.set_ylabel('激活频率', fontsize=12)
ax.set_title('SAE 特征激活频率', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

# 重建质量
ax = axes[1]
with torch.no_grad():
    _, x_hat = sae(activations[:100])
x_orig = activations[:100, 0].numpy()
x_recon = x_hat[:, 0].numpy()
ax.scatter(x_orig, x_recon, alpha=0.5, s=20)
ax.plot([-6, 6], [-6, 6], 'r--', linewidth=2)
ax.set_xlabel('原始激活', fontsize=12)
ax.set_ylabel('重建激活', fontsize=12)
ax.set_title('SAE 重建质量', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_sae.png', bbox_inches='tight')
plt.show()
print("左: 大部分特征稀疏 (低激活频率); 右: 重建接近原始。")""")

md("""## 3. 发现可解释特征

### 3.1 特征解释

```
SAE 发现的特征 (示例):
  - "法国" 特征: 对法国相关 token 激活
  - "代码" 特征: 对代码相关内容激活
  - "数学" 特征: 对数学表达式激活
  - "有害" 特征: 对有害内容激活

→ 每个特征对应一个人类可理解的概念
```

### 3.2 电路发现

```
有了可解释特征，可以找电路:
  1. 找相关特征 (如 "法国" 和 "首都")
  2. 追踪特征间的信息流
  3. 发现 "法国 → 首都 → 巴黎" 电路

→ 理解模型如何做推理
```""")

code("""# 模拟特征解释
feature_names = ['法国', '首都', '代码', '数学', '有害', '中文', '科学', '历史']
np.random.seed(42)

# 模拟特征激活模式
contexts = ['巴黎是法国首都', '写一个Python函数', '勾股定理', '危险内容', '北京是中国的首都']
context_features = {
    '巴黎是法国首都': [0.9, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    '写一个Python函数': [0.0, 0.0, 0.95, 0.1, 0.0, 0.0, 0.0, 0.0],
    '勾股定理': [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0],
    '危险内容': [0.0, 0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0],
    '北京是中国的首都': [0.0, 0.7, 0.0, 0.0, 0.0, 0.9, 0.0, 0.2],
}

print("特征解释 (SAE 发现的可解释特征):")
print("=" * 60)
for context, activations in context_features.items():
    active = [(name, act) for name, act in zip(feature_names, activations) if act > 0.1]
    active.sort(key=lambda x: -x[1])
    print(f"\\n'{context}':")
    for name, act in active:
        print(f"  {name}: {act:.2f}")""")

md("""## 4. 应用

### 4.1 安全监控

```
有了 SAE 特征，可以:
  1. 监控 "有害" 特征 → 检测有害输出
  2. 监控 "欺骗" 特征 → 检测欺骗行为
  3. 监控 "越狱" 特征 → 检测越狱尝试

→ 比黑盒检测更可靠
```

### 4.2 模型调试

```
当模型出错时:
  1. 看哪些特征异常激活
  2. 追踪信息流找问题
  3. 定位到具体电路
  → 精准修复
```""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 机制可解释性 | ✅ |
| 稀疏自编码器 (SAE) | ✅ |
| 多义性 → 单义特征 | ✅ |
| 特征解释与电路发现 | ✅ |

### 核心 takeaway
> **SAE 把多义神经元分解为单义特征**——每个特征对应可理解概念，让黑盒变透明。机制可解释性是 AI 安全的关键。

### 🔗 下一章
**`70_representation_engineering.ipynb`** — RepE/激活工程

---

> 💬 **板块十二(机制可解释性)进行中 (1/3)。**""")

output_path = "notebooks/69_sae_interpretability.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")