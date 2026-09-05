# 生成 70_representation_engineering.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 70 — 表征工程与激活工程

> 🔥 不只理解模型内部，还能"编辑"它——表征工程让 AI 更诚实、更安全。

## 本章你将掌握

1. **表征工程 (RepE)**：在表征空间控制模型
2. **激活工程**：修改激活改变行为
3. **表征控制**：让模型更诚实/无害
4. **vs 微调**：更轻量的行为控制""")

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

md("""## 1. 表征工程 (RepE)

### 1.1 核心思想

```
表征工程 (Representation Engineering):
  在表征空间理解和控制模型

  理解: 找到表征空间中的"概念方向"
    - 诚实方向
    - 有害方向
    - 情感方向

  控制: 沿概念方向调整激活
    - 增加诚实 → 更诚实
    - 减少有害 → 更安全
    - 调整情感 → 改变语气

vs 微调:
  微调: 修改权重 → 训练 → 慢
  RepE: 修改激活 → 推理时 → 快
```

> 💡 RepE 在推理时控制模型行为——不需要重新训练，即时生效。""")

md("""## 2. 激活工程

### 2.1 方法

```
激活工程 (Activation Engineering):
  1. 找概念方向 (如 "诚实")
     - 收集诚实/不诚实回答的激活
     - 差值 → 概念方向

  2. 推理时修改激活
     h' = h + α * concept_direction
     α > 0: 增强
     α < 0: 抑制

  3. 效果
     增强诚实 → 模型更倾向说实话
     抑制有害 → 模型不产生有害内容
```""")

code("""# 激活工程实现
class ActivationEngineer:
    def __init__(self, model_dim=64):
        self.model_dim = model_dim
        self.concept_directions = {}

    def find_concept_direction(self, concept_name, positive_acts, negative_acts):
        # 正例和负例的激活差 → 概念方向
        pos_mean = positive_acts.mean(dim=0)
        neg_mean = negative_acts.mean(dim=0)
        direction = pos_mean - neg_mean
        direction = direction / direction.norm()  # 归一化
        self.concept_directions[concept_name] = direction
        return direction

    def modify_activation(self, h, concept_name, alpha=1.0):
        if concept_name not in self.concept_directions:
            return h
        direction = self.concept_directions[concept_name]
        return h + alpha * direction

    def measure_concept(self, h, concept_name):
        if concept_name not in self.concept_directions:
            return 0.0
        direction = self.concept_directions[concept_name]
        return torch.dot(h, direction).item()

# 模拟: 找"诚实"方向
np.random.seed(42)
model_dim = 64
engineer = ActivationEngineer(model_dim)

# 诚实和不诚实的激活
honest_acts = torch.randn(50, model_dim) + torch.randn(model_dim) * 2
dishonest_acts = torch.randn(50, model_dim) - torch.randn(model_dim) * 2

honest_dir = engineer.find_concept_direction("诚实", honest_acts, dishonest_acts)

# 找"有害"方向
safe_acts = torch.randn(50, model_dim) + torch.randn(model_dim) * 2
harmful_acts = torch.randn(50, model_dim) - torch.randn(model_dim) * 2
harmful_dir = engineer.find_concept_direction("有害", harmful_acts, safe_acts)

print("表征工程:")
print(f"  '诚实' 方向范数: {honest_dir.norm():.4f}")
print(f"  '有害' 方向范数: {harmful_dir.norm():.4f}")
print(f"  两方向正交性: {torch.dot(honest_dir, harmful_dir).item():.4f}")

# 测试: 修改激活
h_test = torch.randn(model_dim)
honest_score = engineer.measure_concept(h_test, "诚实")
harmful_score = engineer.measure_concept(h_test, "有害")

# 增强诚实
h_enhanced = engineer.modify_activation(h_test, "诚实", alpha=2.0)
honest_score_new = engineer.measure_concept(h_enhanced, "诚实")

print(f"\\n激活工程效果:")
print(f"  原始诚实分数: {honest_score:.4f}")
print(f"  增强后诚实分数: {honest_score_new:.4f}")
print(f"  变化: +{honest_score_new - honest_score:.4f}")""")

code("""# 可视化激活工程
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 概念方向
ax = axes[0]
honest_2d = honest_acts[:, :2].numpy()
dishonest_2d = dishonest_acts[:, :2].numpy()
ax.scatter(honest_2d[:, 0], honest_2d[:, 1], c='green', alpha=0.5, label='诚实')
ax.scatter(dishonest_2d[:, 0], dishonest_2d[:, 1], c='red', alpha=0.5, label='不诚实')
dir_2d = honest_dir[:2].numpy()
ax.arrow(0, 0, dir_2d[0]*3, dir_2d[1]*3, head_width=0.1, head_length=0.1,
         fc='blue', ec='blue', linewidth=3, label='诚实方向')
ax.set_title('概念方向: 诚实', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3); ax.set_aspect('equal')

# alpha 效果
ax = axes[1]
alphas = np.linspace(-3, 3, 50)
honest_scores = []
harmful_scores = []
for alpha in alphas:
    h_mod = engineer.modify_activation(h_test, "诚实", alpha=alpha)
    honest_scores.append(engineer.measure_concept(h_mod, "诚实"))
    harmful_scores.append(engineer.measure_concept(h_mod, "有害"))

ax.plot(alphas, honest_scores, 'b-', linewidth=2, label='诚实分数')
ax.plot(alphas, harmful_scores, 'r--', linewidth=2, label='有害分数')
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('α (增强强度)', fontsize=12)
ax.set_ylabel('概念分数', fontsize=12)
ax.set_title('激活工程: 调节 α 控制行为', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_repe.png', bbox_inches='tight')
plt.show()
print("左: 诚实方向区分诚实/不诚实; 右: α越大越诚实。")""")

md("""## 3. 表征控制应用

### 3.1 安全控制

```
应用: 抑制有害
  h' = h - α * harmful_direction
  → 模型不产生有害内容

vs RLHF:
  RLHF: 训练时修改权重
  RepE: 推理时修改激活
  → RepE 更快、更灵活
```

### 3.2 诚实增强

```
应用: 增强诚实
  h' = h + α * honesty_direction
  → 模型更倾向说实话

  如果模型"知道"但"不说"
  → 增强诚实可以让它说出来
```""")

code("""# 表征控制对比
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['微调 (LoRA)', 'RLHF', 'RepE', 'Prompt工程', 'DPO']
speed = [3, 2, 10, 10, 5]
flexibility = [6, 5, 9, 7, 6]
effect = [8, 9, 7, 5, 8]

x = np.arange(len(methods))
width = 0.25

ax.bar(x - width, speed, width, label='速度', color='steelblue', alpha=0.8)
ax.bar(x, flexibility, width, label='灵活性', color='coral', alpha=0.8)
ax.bar(x + width, effect, width, label='效果', color='forestgreen', alpha=0.8)

ax.set_xlabel('方法', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('行为控制方法对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10, rotation=15)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_repe_comparison.png', bbox_inches='tight')
plt.show()
print("RepE 速度和灵活性最优; RLHF 效果最好但最慢。")""")

md("""## 4. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 表征工程 (RepE) | ✅ |
| 激活工程 | ✅ |
| 概念方向发现 | ✅ |
| 行为控制应用 | ✅ |

### 核心 takeaway
> **RepE 在推理时控制模型行为**——找概念方向，调激活强度。比微调更快更灵活，是轻量行为控制的新范式。

### 🔗 下一章
**`71_probing_attribution.ipynb`** — 探针、归因、不确定性量化

---

> 💬 **板块十二(机制可解释性)进行中 (2/3)。**""")

output_path = "notebooks/70_representation_engineering.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")