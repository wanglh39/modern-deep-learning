"""生成 21_inference_time_methods.ipynb 的脚本"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 21 — 推理时方法：best-of-N / self-consistency / verification

> 不改模型，只在推理时多花计算——就能提升效果。
> 这些方法是"免费的"性能提升，适用于任何模型。

## 本章你将掌握

1. **Best-of-N**：生成N个选最好
2. **Self-consistency**：多数投票
3. **Verification**：验证答案再修正
4. **对比**：不同方法的计算-效果权衡""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120; plt.rcParams['font.size'] = 11
np.random.seed(42)
print("环境准备完成 ✅")""")

md("""## 1. Best-of-N：生成N个选最好

### 1.1 方法

```
生成 N 个答案 → 用打分器选最好的
```

打分器可以是：
- **奖励模型**：给每个答案打分
- **PRM**：给推理过程打分
- **验证器**：检查答案是否正确

### 1.2 效果

如果单个答案正确率 $p$，N 个中至少一个正确的概率：

$$P_{best} = 1 - (1-p)^N$$""")

code("""# Best-of-N 理论效果
p = 0.3  # 单次正确率
Ns = [1, 2, 4, 8, 16, 32, 64, 128]
best_of_n = [1 - (1-p)**n for n in Ns]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Ns, best_of_n, 'b-o', linewidth=2.5, markersize=8)
ax.set_xscale('log')
ax.set_xlabel('N (生成次数)')
ax.set_ylabel('至少一个正确的概率')
ax.set_title(f'Best-of-N 效果 (单次正确率={p})')
ax.grid(True, alpha=0.3)
for n, prob in zip(Ns, best_of_n):
    ax.annotate(f'{prob:.1%}', (n, prob), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('notebooks/fig_best_of_n.png', bbox_inches='tight')
plt.show()
print(f"单次正确率{p}: N=1→{p:.0%}, N=8→{1-(1-p)**8:.0%}, N=64→{1-(1-p)**64:.1%}")
print("Best-of-N: 多生成几个, 至少一个对的概率快速上升。")""")

md("""## 2. Self-Consistency：多数投票

### 2.1 方法

```
生成 N 个答案 → 多数投票选最常见的
```

适用于**有唯一正确答案**的问题（数学、逻辑）。

### 2.2 vs Best-of-N

| | Best-of-N | Self-consistency |
|---|-----------|-----------------|
| **需要打分器** | ✅ | ❌ |
| **适用** | 开放生成 | 有唯一答案 |
| **原理** | 选最好的 | 选最常见的 |

> 💡 Self-consistency 不需要打分器——只要多数答案一致，大概率是对的。""")

code("""# Self-consistency 模拟
def generate_answer(correct_prob=0.6, correct_answer=42):
    if np.random.random() < correct_prob:
        return correct_answer
    return np.random.randint(0, 100)  # 随机错误答案

def self_consistency(n_samples=10, correct_prob=0.6):
    answers = [generate_answer(correct_prob) for _ in range(n_samples)]
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common == 42  # 是否选对了

# 不同 N 的效果
Ns = [1, 3, 5, 10, 20, 50, 100]
sc_acc = []
for n in Ns:
    correct = sum(self_consistency(n, correct_prob=0.4) for _ in range(500))
    sc_acc.append(correct / 500)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Ns, sc_acc, 'r-s', linewidth=2.5, markersize=8, label='Self-consistency')
ax.axhline(0.4, color='gray', linestyle='--', alpha=0.5, label='单次正确率')
ax.set_xscale('log')
ax.set_xlabel('N (采样次数)')
ax.set_ylabel('准确率')
ax.set_title('Self-Consistency: 多数投票效果')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_self_consistency.png', bbox_inches='tight')
plt.show()
print("Self-consistency: 即使单次只有40%正确, N=100时准确率大幅提升。")""")

md("""## 3. Verification：验证再修正

### 3.1 方法

```
生成答案 → 验证器检查 → 如果错, 重新生成/修正
```

### 3.2 验证器类型

| 验证器 | 说明 | 例子 |
|--------|------|------|
| **代码执行** | 运行代码看结果 | 代码题 |
| **数学验证** | 代入检查 | 方程求解 |
| **LLM验证** | 另一个LLM检查 | 通用 |
| **PRM** | 过程奖励 | 推理链 |""")

code("""# Verification 模拟
def generate_and_verify(max_attempts=5, correct_prob=0.4):
    # 生成+验证: 错了就重试
    for attempt in range(max_attempts):
        answer = generate_answer(correct_prob)
        if answer == 42:  # 验证通过
            return True, attempt + 1
    return False, max_attempts

# 不同最大尝试次数的效果
max_attempts_list = [1, 2, 3, 5, 10]
verify_acc = []
avg_attempts = []
for max_a in max_attempts_list:
    results = [generate_and_verify(max_a, correct_prob=0.3) for _ in range(500)]
    acc = sum(r[0] for r in results) / 500
    avg = np.mean([r[1] for r in results])
    verify_acc.append(acc)
    avg_attempts.append(avg)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(max_attempts_list, verify_acc, 'g-o', linewidth=2.5, markersize=8)
axes[0].set_xlabel('最大尝试次数')
axes[0].set_ylabel('准确率')
axes[0].set_title('Verification: 重试效果')
axes[0].grid(True, alpha=0.3)

axes[1].plot(max_attempts_list, avg_attempts, 'b-o', linewidth=2.5, markersize=8)
axes[1].set_xlabel('最大尝试次数')
axes[1].set_ylabel('平均尝试次数')
axes[1].set_title('Verification: 计算开销')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_verification.png', bbox_inches='tight')
plt.show()
print("Verification: 允许重试→准确率提升, 但计算开销也增加。")""")

md("""## 4. 方法对比

| 方法 | 需要打分器 | 计算量 | 效果 | 适用 |
|------|-----------|--------|------|------|
| **Best-of-N** | ✅ | N×生成+打分 | 好 | 开放生成 |
| **Self-consistency** | ❌ | N×生成 | 好 | 唯一答案 |
| **Verification** | ✅ | ~N×生成 | 很好 | 可验证问题 |

> 💡 实际中可以组合：Self-consistency + Verification = 先投票再验证。""")

code("""# 综合对比
methods = ['单次', 'Best-of-8', 'SC-8', 'Verify-3', 'SC-8+Verify']
accs = [0.30, 0.70, 0.65, 0.66, 0.78]
costs = [1, 8, 8, 3, 11]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(methods, accs, color=['gray', 'blue', 'red', 'green', 'purple'], alpha=0.7)
axes[0].set_ylabel('准确率'); axes[0].set_title('方法对比: 准确率')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(methods, costs, color=['gray', 'blue', 'red', 'green', 'purple'], alpha=0.7)
axes[1].set_ylabel('计算量 (相对)'); axes[1].set_title('方法对比: 计算开销')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/fig_inference_methods.png', bbox_inches='tight')
plt.show()
print("SC+Verify: 准确率最高但计算也最多——效果和成本的权衡。")""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| Best-of-N | ✅ |
| Self-consistency 多数投票 | ✅ |
| Verification 验证重试 | ✅ |
| 计算-效果权衡 | ✅ |

### 核心 takeaway
> **推理时方法不改模型就能提升效果**——Best-of-N、SC、Verification。
> 它们是"免费"的性能提升，适用于任何模型。

### 🔗 下一章
**`22_long_context.ipynb`** — 长上下文、Ring Attention、位置编码外推""")

output_path = "notebooks/21_inference_time_methods.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")