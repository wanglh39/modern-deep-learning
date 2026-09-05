"""生成 14_data_engineering.ipynb 的脚本"""
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
md("""# 14 — 数据工程：合成、筛选、配比

> "Garbage in, garbage out"——模型再好，数据差也没用。
> 现代 LLM 训练 70% 的功夫在数据上。GPT-4 的数据管线比模型架构更保密。
> 本章覆盖数据工程的三个核心：合成、筛选、配比。

## 本章你将掌握

1. **合成数据**：用模型生成训练数据
2. **质量筛选**：启发式 + 模型打分 + 去重
3. **数据配比**：不同类型数据的混合比例
4. **自训练/自我改进**：模型用自己的输出训练自己""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
print("环境准备完成 ✅")""")

# ============================================================
md("""## 1. 数据工程为什么重要

### 1.1 数据是瓶颈

```
2020: 模型架构是瓶颈 (Transformer 刚出)
2022: 算力是瓶颈 (训练要几千卡)
2024: 数据是瓶颈 (互联网高质量文本快用完了)
```

### 1.2 数据管线

```
原始数据 → 清洗 → 去重 → 质量筛选 → 配比混合 → 训练数据
  ↑                                        ↓
  互联网爬取                          合成数据补充
```

> 💡 Llama 2 的数据管线：2T tokens，经过 20+ 道筛选工序。
> 数据质量比数据量更重要——Chinchilla 定律说的 20N 是**高质量**数据。""")

# ============================================================
md("""## 2. 合成数据

### 2.1 为什么需要合成数据？

- **高质量人类数据快用完了**——互联网文本就那么多
- **特定领域数据稀缺**——数学、代码、推理数据不够
- **合成数据可以针对性生成**

### 2.2 合成数据的方法

| 方法 | 说明 | 应用 |
|------|------|------|
| **指令回译** | 用强模型生成指令-回答对 | SFT 数据 |
| **拒绝采样** | 生成多个，选最好的 | 推理数据 |
| **自我改进** | 模型用自己的输出训练 | 迭代提升 |
| **数据增强** | 改写、翻译、扰动 | 扩充数据量 |

> 💡 GPT-4 大量用合成数据训练——用 GPT-3.5 生成数据，GPT-4 学习。
> DeepSeek-R1 的推理数据也是合成的——用 R1-Zero 生成，再筛选。""")

code("""# 模拟合成数据生成
# 用简单模板 + 随机组合生成"指令-回答"对
templates = [
    ("解释{concept}", "{concept}是指{definition}。"),
    ("什么是{concept}？", "{concept}是{definition}的概念。"),
    ("举例说明{concept}", "例如：{example}。这就是{concept}。"),
]

concepts = {
    "梯度下降": {"definition": "沿梯度反方向更新参数的优化算法", "example": "w = w - lr * gradient"},
    "反向传播": {"definition": "通过链式法则计算梯度的算法", "example": "从输出层向输入层逐层计算梯度"},
    "注意力机制": {"definition": "让模型关注输入中重要部分的机制", "example": "Transformer 中的 self-attention"},
    "过拟合": {"definition": "模型在训练集上表现好但泛化差", "example": "训练准确率99%但测试只有70%"},
}

def generate_synthetic_data(n=20):
    \"\"\"合成指令-回答数据\"\"\"
    data = []
    for _ in range(n):
        concept = np.random.choice(list(concepts.keys()))
        template = templates[np.random.randint(len(templates))]
        instruction = template[0].format(concept=concept)
        response = template[1].format(
            concept=concept,
            definition=concepts[concept]["definition"],
            example=concepts[concept]["example"]
        )
        data.append({"instruction": instruction, "response": response})
    return data

synthetic_data = generate_synthetic_data(10)
print("合成数据示例:")
for i, d in enumerate(synthetic_data[:3]):
    print(f"\\n  [{i+1}] 指令: {d['instruction']}")
    print(f"      回答: {d['response']}")

print(f"\\n共生成 {len(synthetic_data)} 条合成数据——成本远低于人工标注。")""")

# ============================================================
md("""## 3. 质量筛选

### 3.1 筛选管线

```
原始数据
  → 1. 启发式过滤 (长度/语言/重复度)
  → 2. 安全过滤 (有害内容)
  → 3. 模型打分 (质量分类器)
  → 4. 去重 (MinHash/精确去重)
  → 高质量数据
```

### 3.2 去重的重要性

重复数据会让模型**过拟合**到重复内容，浪费训练容量。
研究表明，去重后用**更少数据**能达到**更好效果**。""")

code("""# 模拟数据筛选管线
raw_data = [
    "The gradient descent algorithm updates parameters iteratively.",
    "The gradient descent algorithm updates parameters iteratively.",  # 重复
    "a b c d e",  # 太短
    "The attention mechanism allows models to focus on relevant parts.",
    "The attention mechanism allows models to focus on relevant parts.",  # 重复
    "This is a high quality explanation of backpropagation in neural networks, covering the chain rule and computational graphs.",
    "asdf jkl qwer",  # 乱码
    "Gradient descent is an optimization algorithm used in machine learning to minimize the loss function by iteratively moving towards the steepest descent.",
    "The The The The The The The",  # 高重复度
    "Neural networks learn representations through backpropagation and gradient descent, which are fundamental to deep learning.",
]

def heuristic_filter(text, min_len=10, max_repeat_ratio=0.3):
    \"\"\"启发式过滤\"\"\"
    # 长度过滤
    if len(text.split()) < min_len:
        return False, "太短"
    # 重复度过滤
    words = text.split()
    word_counts = Counter(words)
    max_repeat = max(word_counts.values()) / len(words)
    if max_repeat > max_repeat_ratio:
        return False, "重复度高"
    return True, "通过"

def quality_score(text):
    \"\"\"简化质量打分 (真实用分类器)\"\"\"
    score = 0
    # 长度适中加分
    if 10 < len(text.split()) < 50:
        score += 0.3
    # 有标点加分
    if '.' in text or ',' in text:
        score += 0.2
    # 词汇多样性加分
    words = text.split()
    unique_ratio = len(set(words)) / len(words)
    score += unique_ratio * 0.5
    return score

def deduplicate(data):
    \"\"\"精确去重\"\"\"
    seen = set()
    result = []
    for text in data:
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result

# 执行筛选管线
print("数据筛选管线:")
print(f"  原始数据: {len(raw_data)} 条")

# 1. 启发式过滤
filtered = []
for text in raw_data:
    passed, reason = heuristic_filter(text)
    if passed:
        filtered.append(text)
print(f"  启发式过滤后: {len(filtered)} 条")

# 2. 去重
deduped = deduplicate(filtered)
print(f"  去重后: {len(deduped)} 条")

# 3. 质量打分
scored = [(text, quality_score(text)) for text in deduped]
scored.sort(key=lambda x: -x[1])

print(f"\\n质量打分 (排序后):")
for text, score in scored:
    print(f"  [{score:.3f}] {text[:60]}...")""")

code("""# 可视化筛选管线
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 管线各阶段数据量
stages = ['原始', '启发式过滤', '去重', '质量筛选']
counts = [len(raw_data), len(filtered), len(deduped), len([s for s in scored if s[1] > 0.5])]
axes[0].bar(stages, counts, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
axes[0].set_ylabel('数据量')
axes[0].set_title('数据筛选管线: 各阶段数据量')
axes[0].grid(True, alpha=0.3, axis='y')

# 质量分数分布
scores = [quality_score(text) for text in deduped]
axes[1].hist(scores, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='质量阈值')
axes[1].set_xlabel('质量分数')
axes[1].set_ylabel('数据条数')
axes[1].set_title('质量分数分布')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_data_pipeline.png', bbox_inches='tight')
plt.show()
print("数据管线: 10条 → 启发式过滤 → 去重 → 质量筛选 → 最终高质量数据。")""")

# ============================================================
md("""## 4. 数据配比

### 4.1 不同类型数据的混合

训练数据通常混合多种类型：

| 数据类型 | Llama 2 比例 | 作用 |
|---------|-------------|------|
| 网页文本 | ~80% | 通用语言能力 |
| 代码 | ~5% | 逻辑推理 |
| 数学 | ~4% | 数学推理 |
| 书籍 | ~4% | 长文本理解 |
| 论文 | ~2% | 学术知识 |
| 对话 | ~5% | 对话能力 |

### 4.2 配比的影响

配比直接影响模型能力：
- **代码多** → 编程强但聊天弱
- **数学多** → 推理强但创意弱
- **对话多** → 聊天好但知识浅

> 💡 配比是**实验密集型**的工程——没有理论指导，靠 ablation study 调。""")

code("""# 模拟不同配比下的模型能力
np.random.seed(42)

# 数据类型
data_types = ['网页文本', '代码', '数学', '书籍', '对话']
# 能力维度
abilities = ['语言', '编程', '推理', '知识', '对话']

# 每种数据类型对各能力的贡献矩阵
contribution = np.array([
    [0.8, 0.1, 0.2, 0.6, 0.4],  # 网页文本
    [0.3, 0.9, 0.6, 0.3, 0.2],  # 代码
    [0.2, 0.3, 0.9, 0.5, 0.1],  # 数学
    [0.6, 0.1, 0.3, 0.8, 0.3],  # 书籍
    [0.5, 0.1, 0.1, 0.2, 0.9],  # 对话
])

# 不同配比方案
mixes = {
    'Llama 2': [0.80, 0.05, 0.04, 0.04, 0.07],
    '代码优先': [0.50, 0.25, 0.10, 0.05, 0.10],
    '数学优先': [0.50, 0.10, 0.25, 0.05, 0.10],
    '对话优先': [0.50, 0.05, 0.05, 0.10, 0.30],
}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(abilities))
width = 0.2

for i, (name, mix) in enumerate(mixes.items()):
    # 能力 = 数据配比 @ 贡献矩阵
    capability = np.array(mix) @ contribution
    ax.bar(x + i*width, capability, width, label=name, alpha=0.8)

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(abilities)
ax.set_ylabel('能力分数')
ax.set_title('不同数据配比下的模型能力')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('notebooks/fig_data_mix.png', bbox_inches='tight')
plt.show()
print("配比直接影响能力: 代码优先→编程强, 数学优先→推理强, 对话优先→对话强。")""")

# ============================================================
md("""## 5. 自训练与自我改进

### 5.1 自训练（Self-Training）

用模型自己的预测作为训练数据：

```
1. 用初始模型在无标注数据上预测
2. 把高置信度预测当作伪标签
3. 用伪标签数据训练新模型
4. 重复
```

### 5.2 自我改进（Self-Improvement）

让模型**生成**训练数据而非只是预测：

```
1. 给模型指令，让它生成回答
2. 用奖励模型/过滤器选好的回答
3. 用这些 (指令, 好回答) 训练模型
4. 重复 → 模型越来越好
```

> 💡 DeepSeek-R1-Zero 就是纯自我改进——直接在 RL 阶段让模型自己探索推理策略。
> 这产生了"aha moment"等涌现行为。""")

code("""# 模拟自我改进过程
class SimpleSelfImprovement:
    \"\"\"模拟自我改进: 生成→筛选→训练→重复\"\"\"
    def __init__(self, initial_quality=0.3):
        self.quality = initial_quality
        self.history = [initial_quality]

    def generate(self):
        \"\"\"生成数据, 质量随模型质量提升\"\"\"
        return self.quality + np.random.randn(20) * 0.1

    def filter(self, generated, threshold=None):
        \"\"\"筛选: 只保留高于阈值的数据\"\"\"
        if threshold is None:
            threshold = np.percentile(generated, 70)  # 取top 30%
        return [g for g in generated if g > threshold]

    def train(self, filtered_data):
        \"\"\"用筛选后数据训练, 质量提升\"\"\"
        if len(filtered_data) > 0:
            improvement = np.mean(filtered_data) * 0.1
            self.quality = 0.9 * self.quality + 0.1 * (self.quality + improvement)
        self.history.append(self.quality)

# 运行自我改进
np.random.seed(42)
si = SimpleSelfImprovement(initial_quality=0.3)

for iteration in range(20):
    generated = si.generate()
    filtered = si.filter(generated)
    si.train(filtered)

print("自我改进过程:")
for i in [0, 5, 10, 15, 20]:
    if i < len(si.history):
        print(f"  迭代 {i:2d}: 模型质量 = {si.history[i]:.3f}")

# 对比: 无筛选的自训练 (可能退化)
np.random.seed(42)
si_no_filter = SimpleSelfImprovement(initial_quality=0.3)
for iteration in range(20):
    generated = si_no_filter.generate()
    # 不筛选, 用所有数据
    si_no_filter.train(generated)

print(f"\\n有筛选最终质量: {si.history[-1]:.3f}")
print(f"无筛选最终质量: {si_no_filter.history[-1]:.3f}")
print("筛选是自我改进的关键——只用高质量数据训练才能持续提升。")""")

code("""# 可视化自我改进过程
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(si.history, 'b-o', linewidth=2.5, markersize=6, label='有筛选 (自我改进)')
ax.plot(si_no_filter.history, 'r-s', linewidth=2.5, markersize=6, label='无筛选 (可能退化)')
ax.set_xlabel('迭代轮次')
ax.set_ylabel('模型质量')
ax.set_title('自我改进: 筛选 vs 无筛选')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('notebooks/fig_self_improvement.png', bbox_inches='tight')
plt.show()
print("有筛选的自我改进持续提升; 无筛选可能退化——数据质量是关键。")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 合成数据（指令回译/拒绝采样） | ✅ |
| 质量筛选管线（启发式+去重+打分） | ✅ |
| 数据配比对模型能力的影响 | ✅ |
| 自训练与自我改进 | ✅ |
| 筛选在自我改进中的关键作用 | ✅ |

### 核心 takeaway

> **数据工程是现代 LLM 的核心竞争力**。
> 合成数据解决数据枯竭，筛选保证质量，配比决定能力方向。
> 自我改进 + 筛选 = 持续提升的飞轮。

### 🔗 下一章预告

**`15_training_techniques.ipynb`** — 学习率调度/warmup/cosine/梯度累积/裁剪

---

> 💬 **写在最后**：模型架构可以公开，但数据管线是真正的护城河。
> 理解数据工程，就理解了为什么同样架构的模型效果差很多。""")

# ============================================================
output_path = "notebooks/14_data_engineering.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")