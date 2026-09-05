"""生成 20_prm_process_reward.ipynb 的脚本"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 20 — PRM：过程奖励模型

> 普通奖励模型只给**最终答案**打分。PRM (Process Reward Model) 给**每一步推理**打分。
> 这让 RL 能在中间步骤就获得反馈——不必等到最后才知道对错。

## 本章你将掌握

1. **ORM vs PRM**：结果奖励 vs 过程奖励
2. **PRM 训练**：从步骤标注学习
3. **PRM 在推理模型中的作用**
4. **自动标注**：MCTS + PRM 自举""")

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

md("""## 1. ORM vs PRM

### 1.1 两种奖励模型

```
ORM (Outcome Reward Model):
  问题 → 推理步骤1 → 步骤2 → 步骤3 → 答案 → 奖励(答案对不对)
  只有最终一个标量奖励

PRM (Process Reward Model):
  问题 → 步骤1[奖励1] → 步骤2[奖励2] → 步骤3[奖励3] → 答案[奖励4]
  每步都有奖励
```

### 1.2 PRM 的优势

| | ORM | PRM |
|---|-----|-----|
| **反馈粒度** | 只有最终 | 每一步 |
| **信用分配** | 难 (最后奖励要分给所有步骤) | 易 (每步直接有奖励) |
| **错误定位** | 不知道哪步错 | 精确知道 |
| **训练效率** | 低 | 高 |

> 💡 PRM 让 RL 训练更高效——不用等到最后才知道对错，中间步骤就能反馈。
> OpenAI 的 "Let's Verify Step by Step" 证明 PRM 比 ORM 在数学推理上好很多。""")

code("""# PRM vs ORM 演示
class SimplePRM(nn.Module):
    # 过程奖励模型: 每步打分
    def __init__(self, d=16):
        super().__init__()
        self.embed = nn.Embedding(100, d)
        self.fc = nn.Linear(d, 1)
    def forward(self, step_id):
        return self.fc(self.embed(step_id)).squeeze(-1)

class SimpleORM(nn.Module):
    # 结果奖励模型: 只给最终答案打分
    def __init__(self, d=16):
        super().__init__()
        self.embed = nn.Embedding(100, d)
        self.fc = nn.Linear(d, 1)
    def forward(self, answer_id):
        return self.fc(self.embed(answer_id)).squeeze(-1)

# 模拟一个推理过程: 5步
# 步骤: [正确, 正确, 错误, 正确, 正确] → 最终答案错
steps = [10, 11, 12, 13, 14]  # 步骤token ids
step_labels = [1, 1, 0, 1, 1]  # 每步是否正确 (第3步错)
answer = 20  # 最终答案 (因为第3步错, 答案错)
answer_label = 0

# PRM: 训练每步打分
prm = SimplePRM()
opt_prm = torch.optim.Adam(prm.parameters(), lr=1e-2)
for _ in range(100):
    loss = 0
    for sid, label in zip(steps, step_labels):
        pred = prm(torch.tensor(sid))
        loss += F.binary_cross_entropy_with_logits(pred, torch.tensor(float(label)))
    opt_prm.zero_grad(); loss.backward(); opt_prm.step()

# ORM: 只训练最终答案
orm = SimpleORM()
opt_orm = torch.optim.Adam(orm.parameters(), lr=1e-2)
for _ in range(100):
    pred = orm(torch.tensor(answer))
    loss = F.binary_cross_entropy_with_logits(pred, torch.tensor(float(answer_label)))
    opt_orm.zero_grad(); loss.backward(); opt_orm.step()

print("PRM (过程奖励) 打分:")
for i, sid in enumerate(steps):
    with torch.no_grad():
        score = torch.sigmoid(prm(torch.tensor(sid))).item()
    status = "✅" if step_labels[i] else "❌"
    print(f"  步骤{i+1}: score={score:.3f} {status}")

with torch.no_grad():
    orm_score = torch.sigmoid(orm(torch.tensor(answer))).item()
print(f"\\nORM (结果奖励) 打分: {orm_score:.3f} (只知道最终错, 不知道哪步错)")
print("\\nPRM 精确定位第3步错误; ORM只知道最终答案错。")""")

md("""## 2. PRM 在推理模型中的作用

### 2.1 用 PRM 做 RL 训练

```
生成推理链: 步骤1 → 步骤2 → ... → 步骤N
PRM 打分:   r1    → r2    → ... → rN
总奖励: R = r1 + r2 + ... + rN  (每步都贡献)
RL 更新: 用 R 优化策略
```

### 2.2 Best-of-N with PRM

用 PRM 从 N 个推理链中选最好的：

```
生成 N 个推理链 → PRM 给每个链的每步打分 → 选总分最高的
```

> 💡 OpenAI 论文：PRM + Best-of-N 在 MATH 上达到 78.2%，远超 ORM 的 72.4%。""")

code("""# Best-of-N with PRM
def generate_reasoning_chain(n_steps=5, quality=0.7):
    # 生成推理链, 每步有概率正确
    return [1 if np.random.random() < quality else 0 for _ in range(n_steps)]

def prm_score(chain, prm_model):
    # PRM 给推理链打分 (简化: 正确步骤越多分越高)
    return sum(chain) / len(chain)

def best_of_n_with_prm(n=8, n_steps=5, quality=0.6):
    # 生成N个链, 用PRM选最好的
    chains = [generate_reasoning_chain(n_steps, quality) for _ in range(n)]
    scores = [prm_score(c, None) for c in chains]
    best_idx = np.argmax(scores)
    return chains[best_idx]

def best_of_n_with_orm(n=8, n_steps=5, quality=0.6):
    # ORM: 只看最终答案 (全对才算对)
    chains = [generate_reasoning_chain(n_steps, quality) for _ in range(n)]
    # ORM 只能看最终结果: 全对=1, 否则=0
    results = [all(c) for c in chains]
    if any(results):
        return [1] * n_steps  # 选了一个全对的
    return chains[0]  # 没有全对的, 随机选

# 对比
n_trials = 1000
prm_correct = 0
orm_correct = 0
for _ in range(n_trials):
    prm_result = best_of_n_with_prm(n=8, n_steps=5, quality=0.6)
    orm_result = best_of_n_with_orm(n=8, n_steps=5, quality=0.6)
    if all(prm_result): prm_correct += 1
    if all(orm_result): orm_correct += 1

print(f"Best-of-8 对比 ({n_trials}次试验):")
print(f"  ORM (只看结果): {orm_correct/n_trials:.1%}")
print(f"  PRM (看每步):   {prm_correct/n_trials:.1%}")
print("PRM 能选出部分正确的链; ORM 只能选全对或随机。")""")

md("""## 3. PRM 的自动标注

### 3.1 人工标注步骤太贵

给每步推理打分需要大量人工标注。解法：**自动标注**。

### 3.2 MCTS + PRM 自举

```
1. 用模型生成推理链 + MCTS搜索
2. 用当前PRM给节点打分
3. MCTS找到正确路径 → 自动获得步骤标签
4. 用标签训练更好的PRM
5. 重复 → PRM越来越好
```

> 💡 这就是 "Let's Verify Step by Step" 的方法——用 MCTS 自动生成步骤标签，不需要人工。""")

code("""# 模拟 MCTS + PRM 自举
np.random.seed(42)
prm_quality = 0.5  # 初始PRM质量
history = [prm_quality]

for iteration in range(20):
    # 1. 用当前PRM指导MCTS搜索
    # 简化: PRM质量越高, 搜索越有效
    search_quality = prm_quality * 0.7 + 0.3

    # 2. MCTS找到正确路径 → 生成标签
    # 简化: 搜索质量越高, 标签越多越准
    n_labels = int(search_quality * 100)
    label_accuracy = search_quality

    # 3. 用标签训练PRM
    # 简化: PRM质量提升
    prm_quality = 0.8 * prm_quality + 0.2 * (search_quality + 0.1)
    history.append(prm_quality)

print("MCTS + PRM 自举:")
for i in [0, 5, 10, 15, 20]:
    if i < len(history):
        print(f"  迭代{i:2d}: PRM质量 = {history[i]:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history, 'b-o', linewidth=2.5, markersize=6)
ax.set_xlabel('自举迭代')
ax.set_ylabel('PRM 质量')
ax.set_title('MCTS + PRM 自举: 自动标注提升PRM')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_prm_bootstrap.png', bbox_inches='tight')
plt.show()
print("PRM 通过自举自动提升——不需要人工逐步标注。")""")

md("""## 4. 小结与延伸

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| ORM vs PRM | ✅ |
| PRM 每步打分 | ✅ |
| Best-of-N with PRM | ✅ |
| MCTS + PRM 自举 | ✅ |

### 核心 takeaway
> **PRM 给推理每步打分，比只看结果的 ORM 更高效**。
> MCTS 自举让 PRM 不需要人工标注。PRM 是推理模型的关键组件。

### 🔗 下一章
**`21_inference_time_methods.ipynb`** — best-of-N/self-consistency/verification""")

output_path = "notebooks/20_prm_process_reward.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")