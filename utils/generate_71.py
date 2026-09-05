# 生成 71_probing_attribution.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 71 — 探针、归因与不确定性量化

> 理解模型的三把钥匙：探针看内部、归因看输入、不确定性看信心。

## 本章你将掌握

1. **探针 (Probing)**：检测内部表征
2. **归因 (Attribution)**：哪些输入重要
3. **不确定性量化**：模型有多确信
4. **综合应用**：可解释 AI 工具箱""")

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

md("""## 1. 探针 (Probing)

### 1.1 什么是探针？

```
探针 (Probing):
  训练一个小模型检测内部表征包含什么信息

  1. 取模型中间层激活
  2. 冻结激活
  3. 训练探针 (简单分类器)
  4. 探针准确率 → 表征包含多少信息

问题: 探针高准确率 = 模型用了这个信息?
  → 不一定，可能只是包含但不用
  → 需要"因果探针"验证
```""")

code("""# 探针实现
class ProbingClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

# 模拟模型内部激活
np.random.seed(42)
n_samples = 500
hidden_dim = 128
n_classes = 5

# 模拟: 激活包含部分类别信息
labels = torch.randint(0, n_classes, (n_samples,))
activations = torch.randn(n_samples, hidden_dim)
activations[range(n_samples), labels] += 2.0  # 类别信息编码在激活中

# 训练探针
probe = ProbingClassifier(hidden_dim, n_classes)
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

for epoch in range(200):
    logits = probe(activations)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 评估
with torch.no_grad():
    preds = probe(activations).argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()

print(f"探针分析:")
print(f"  激活维度: {hidden_dim}")
print(f"  探针准确率: {accuracy:.2%}")
print(f"  → 表征包含 {accuracy:.0%} 的类别信息")""")

md("""## 2. 归因 (Attribution)

### 2.1 哪些输入重要？

```
归因分析:
  输出 → 哪些输入特征贡献了多少?

方法:
  1. 梯度归因: ∂y/∂x_i
  2. Integrated Gradients: 路径积分
  3. SHAP: 博弈论归因
  4. LIME: 局部线性近似
```""")

code("""# 归因分析
class GradientAttribution:
    def __init__(self, model):
        self.model = model

    def attribute(self, x, target):
        x.requires_grad = True
        output = self.model(x)
        loss = output[0, target]
        loss.backward()
        return x.grad.clone()

    def integrated_gradients(self, x, target, n_steps=50):
        # Integrated Gradients
        baseline = torch.zeros_like(x)
        alphas = torch.linspace(0, 1, n_steps)

        total_grad = torch.zeros_like(x)
        for alpha in alphas:
            x_interp = baseline + alpha * (x - baseline)
            x_interp.requires_grad = True
            output = self.model(x_interp)
            loss = output[0, target]
            self.model.zero_grad()
            loss.backward()
            total_grad += x_interp.grad

        return (x - baseline) * total_grad / n_steps

# 简单模型
simple_model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 5)
)

x_input = torch.randn(1, 10)
target_class = 2

attributor = GradientAttribution(simple_model)
grad_attr = attributor.attribute(x_input.clone(), target_class)
ig_attr = attributor.integrated_gradients(x_input.clone(), target_class)

print("归因分析:")
print(f"  输入: {x_input[0].numpy().round(3)}")
print(f"  梯度归因: {grad_attr[0].numpy().round(3)}")
print(f"  IG 归因:   {ig_attr[0].numpy().round(3)}")
print(f"  → 正值=正向贡献, 负值=负向贡献")""")

code("""# 归因可视化
fig, ax = plt.subplots(figsize=(10, 5))

features = [f'x{i}' for i in range(10)]
grad_values = grad_attr[0].numpy()
ig_values = ig_attr[0].numpy()

x = np.arange(len(features))
width = 0.35

ax.bar(x - width/2, grad_values, width, label='梯度归因', color='steelblue', alpha=0.8)
ax.bar(x + width/2, ig_values, width, label='Integrated Gradients', color='coral', alpha=0.8)

ax.set_xlabel('输入特征', fontsize=12)
ax.set_ylabel('归因值', fontsize=12)
ax.set_title('特征归因对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(features)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig('notebooks/fig_attribution.png', bbox_inches='tight')
plt.show()
print("不同特征对输出的贡献不同 → 理解模型依赖哪些输入。")""")

md("""## 3. 不确定性量化

### 3.1 不确定性来源

```
1. 认知不确定性 (Epistemic)
   - 模型不确定 (没学过)
   - 可以通过数据减少

2. 偶然不确定性 (Aleatoric)
   - 数据本身噪声
   - 无法减少

3. 分布外 (OOD)
   - 输入超出训练分布
   - 不确定性应高
```""")

code("""# 不确定性量化
def entropy_uncertainty(logits):
    # 用熵衡量不确定性
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    return entropy

# 模拟不同情况
np.random.seed(42)

# 确信的预测
confident_logits = torch.tensor([[10, -5, -5, -5, -5]], dtype=torch.float32)
# 不确信的预测
uncertain_logits = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.float32)
# 中等确信
medium_logits = torch.tensor([[3, 2, 1, -1, -2]], dtype=torch.float32)

print("不确定性量化 (熵):")
print(f"  确信预测: 熵 = {entropy_uncertainty(confident_logits).item():.4f}")
print(f"  中等预测: 熵 = {entropy_uncertainty(medium_logits).item():.4f}")
print(f"  不确信:   熵 = {entropy_uncertainty(uncertain_logits).item():.4f}")
print("  → 熵越高越不确定")""")

md("""## 4. 综合：可解释 AI 工具箱

### 4.1 方法对比

```
方法          回答什么问题      适用
探针          内部包含什么?    表征分析
归因          哪些输入重要?    输入解释
不确定性      模型多确信?      置信度
SAE          内部如何编码?    机制理解
RepE         如何控制行为?    行为编辑
注意力        哪里关注?        粗略解释
```""")

code("""# 可解释 AI 工具箱对比
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['探针', '归因', '不确定性', 'SAE', 'RepE', '注意力']
depth = [5, 4, 3, 9, 8, 3]  # 理解深度
cost = [3, 4, 2, 8, 6, 1]   # 计算成本
actionable = [3, 5, 7, 6, 9, 2]  # 可操作性

x = np.arange(len(methods))
width = 0.25

ax.bar(x - width, depth, width, label='理解深度', color='steelblue', alpha=0.8)
ax.bar(x, cost, width, label='计算成本', color='coral', alpha=0.8)
ax.bar(x + width, actionable, width, label='可操作性', color='forestgreen', alpha=0.8)

ax.set_xlabel('方法', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('可解释 AI 工具箱', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_xai_toolbox.png', bbox_inches='tight')
plt.show()
print("SAE 理解最深; RepE 最可操作; 注意力最便宜但最浅。")""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 探针 (检测内部信息) | ✅ |
| 归因 (梯度/IG) | ✅ |
| 不确定性量化 (熵) | ✅ |
| 可解释 AI 工具箱 | ✅ |

### 核心 takeaway
> **探针看内部、归因看输入、不确定性看信心**——三者组合是可解释 AI 的基础工具箱。深度理解用 SAE，行为控制用 RepE。

### 🔗 下一板块
**`72_deep_rl_frontiers.ipynb`** — AlphaGo/AlphaZero、多智能体RL（进入拓展章）

---

> 💬 **板块十二(机制可解释性)100%完成 (3/3)。**""")

output_path = "notebooks/71_probing_attribution.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")