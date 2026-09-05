# 生成 51_evaluation.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 51 — 评估与 Benchmark 设计

> 没有好的评估，就没有好的模型。但评估 LLM 比训练 LLM 更难。

## 本章你将掌握

1. **Benchmark 设计原则**：什么才是好 benchmark？
2. **主要 LLM Benchmark**：MMLU/HumanEval/GSM8K 等
3. **评估挑战**：污染、博弈、过拟合
4. **LLM-as-Judge**：用 LLM 评估 LLM
5. **生产评估**：A/B 测试、在线指标""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
torch.manual_seed(42)
print("环境准备完成 ✅")""")

md("""## 1. 为什么评估很难？

### 1.1 评估的挑战

```
1. 开放性
   - LLM 输出是开放文本
   - 没有"标准答案"
   - 同一问题多种正确回答

2. 能力交织
   - 一个回答同时涉及推理、知识、语言
   - 难以定位是哪种能力不足

3. 污染 (Contamination)
   - benchmark 数据进入训练集
   - 模型"背答案"而非"会做"

4. 博弈 (Gaming)
   - 模型学会针对 benchmark 优化
   - 分数高但能力不真实

5. 人类偏好不一致
   - 不同人对同一回答评分不同
   - 标注噪声大
```

> 💡 "评估 LLM 比训练 LLM 更难"——因为开放性、能力交织、污染等问题。""")

md("""## 2. 主要 LLM Benchmark

### 2.1 知识与语言

```
MMLU (Massive Multitask Language Understanding):
  - 57 个学科的选择题
  - 从初等数学到专业法律
  - 评估"知识广度"

BBH (BIG-Bench Hard):
  - 23 个难题
  - 推理、逻辑、常识
  - 人类也觉得难

HellaSwag:
  - 常识推理
  - 选择最合理的续写
```

### 2.2 代码

```
HumanEval:
  - 164 个 Python 编程题
  - 从函数签名 + 文档生成实现
  - 评估: pass@k (k 次尝试中至少 1 次通过)

MBPP:
  - 974 个基础编程题
  - 更多样化

SWE-bench:
  - 真实 GitHub issue 修复
  - 端到端软件工程能力
```

### 2.3 数学推理

```
GSM8K:
  - 8500 个小学数学应用题
  - 评估数学推理链

MATH:
  - 竞赛数学题
  - 更难，需要形式推理

AIME / Olympiad:
  - 奥林匹克数学
  - 极限测试
```""")

code("""# 主要 Benchmark 可视化
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 知识与语言
ax = axes[0]
benchmarks_knowledge = ['MMLU', 'BBH', 'HellaSwag', 'ARC', 'TruthfulQA']
scores = [72, 65, 85, 80, 55]
colors = ['steelblue'] * 5
ax.barh(benchmarks_knowledge, scores, color=colors, alpha=0.8)
ax.set_xlabel('分数 (%)', fontsize=11)
ax.set_title('知识与语言', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 代码
ax = axes[1]
benchmarks_code = ['HumanEval', 'MBPP', 'SWE-bench', 'APPS', 'CodeContests']
scores = [70, 75, 15, 40, 25]
colors = ['forestgreen'] * 5
ax.barh(benchmarks_code, scores, color=colors, alpha=0.8)
ax.set_xlabel('分数 (%)', fontsize=11)
ax.set_title('代码', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 数学
ax = axes[2]
benchmarks_math = ['GSM8K', 'MATH', 'AIME', 'Olympiad', 'CMATH']
scores = [85, 50, 10, 5, 60]
colors = ['coral'] * 5
ax.barh(benchmarks_math, scores, color=colors, alpha=0.8)
ax.set_xlabel('分数 (%)', fontsize=11)
ax.set_title('数学推理', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_benchmarks.png', bbox_inches='tight')
plt.show()
print("不同 benchmark 评估不同能力: 知识、代码、数学推理。")""")

md("""## 3. Benchmark 设计原则

### 3.1 好 Benchmark 的标准

```
1. 有效性 (Validity)
   - 测量它声称测量的能力
   - 不是测量记忆或模式匹配

2. 可靠性 (Reliability)
   - 不同时间、不同人评估一致
   - 可复现

3. 区分度 (Discriminative Power)
   - 能区分不同水平的模型
   - 不是太简单 (都100%) 也不是太难 (都0%)

4. 抗污染 (Contamination Resistance)
   - 不容易被"背答案"
   - 动态更新

5. 覆盖度 (Coverage)
   - 覆盖目标能力的各个方面
   - 没有盲区
```

### 3.2 动态 Benchmark

```
静态 benchmark 的问题:
  - 发布后模型训练时可能混入
  - 分数虚高

动态 benchmark 解决方案:
  1. 定期更新题目
  2. 私有评测集 (不公开)
  3. 生成新题 (LLM 生成 + 人类验证)
  4. LiveBench: 每月新题

→ 让"背答案"不可能
```

> 💡 动态 benchmark 是对抗污染的关键——定期更新让模型无法记忆。""")

code("""# 污染检测模拟
np.random.seed(42)

n_questions = 1000
n_models = 5

# 无污染: 模型真实能力
true_abilities = np.random.uniform(30, 80, n_models)

# 污染: 部分题目被记忆
contamination_rates = [0, 0.1, 0.2, 0.3, 0.5]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(n_models)
width = 0.15

for i, rate in enumerate(contamination_rates):
    # 污染后的分数 = 真实能力 + 污染题的"免费"分数
    polluted_scores = true_abilities + rate * (100 - true_abilities)
    offset = (i - 2) * width
    ax.bar(x + offset, polluted_scores, width, label=f'污染率 {rate:.0%}',
           alpha=0.85)

ax.set_xlabel('模型', fontsize=12)
ax.set_ylabel('Benchmark 分数', fontsize=12)
ax.set_title('数据污染对 Benchmark 分数的影响', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'模型 {i+1}' for i in range(n_models)])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_contamination.png', bbox_inches='tight')
plt.show()
print("污染率越高，分数越虚高 → 需要动态 benchmark 防污染。")""")

md("""## 4. LLM-as-Judge：用 LLM 评估 LLM

### 4.1 核心思想

```
传统评估:
  人类标注 (A > B) → 昂贵、慢

LLM-as-Judge:
  用强大的 LLM (如 GPT-4) 判断 (A > B)
  → 便宜、快、可扩展

流程:
  1. 给 judge 模型 prompt + 回答A + 回答B
  2. 让 judge 判断哪个更好
  3. 输出偏好或分数
```

### 4.2 优缺点

```
优点:
  - 成本低 (API 调用 vs 人工)
  - 速度快
  - 可解释 (judge 给出理由)
  - 可扩展

缺点:
  - 偏好偏差 (judge 偏好长回答、特定风格)
  - 位置偏差 (先 A 后 B vs 先 B 后 A)
  - 自我偏好 (judge 偏好同族模型)
  - 不能替代人类 (最终需要人类验证)
```

> 💡 LLM-as-Judge 大幅降低评估成本，但有偏差——需要校准和人类验证。""")

code("""# LLM-as-Judge 模拟
class LLMJudge:
    def __init__(self, bias_length=0.0, bias_position=0.0):
        self.bias_length = bias_length  # 偏好长回答
        self.bias_position = bias_position  # 位置偏差

    def judge(self, prompt, response_a, response_b):
        # 模拟 judge 评分
        score_a = self._score(response_a) + self.bias_position
        score_b = self._score(response_b) - self.bias_position

        # 长度偏差
        if len(response_b) > len(response_a):
            score_b += self.bias_length
        else:
            score_a += self.bias_length

        if score_a > score_b:
            return 'A', score_a, score_b
        else:
            return 'B', score_a, score_b

    def _score(self, response):
        # 模拟质量评分
        score = 0.5
        if '正确' in response or '准确' in response:
            score += 0.2
        if '详细' in response:
            score += 0.1
        return score + np.random.randn() * 0.05

# 测试位置偏差
judge = LLMJudge(bias_position=0.1)
prompt = "解释什么是梯度下降"

response_good = "梯度下降是优化算法，沿负梯度方向更新参数。"
response_bad = "不知道。"

# A 在前 vs B 在前
print("位置偏差测试:")
winner1, sa1, sb1 = judge.judge(prompt, response_good, response_bad)
print(f"  好在前: 判 {winner1} (好={sa1:.2f}, 坏={sb1:.2f})")

winner2, sa2, sb2 = judge.judge(prompt, response_bad, response_good)
print(f"  坏在前: 判 {winner2} (坏={sa2:.2f}, 好={sb2:.2f})")

# 测试长度偏差
print("\\n长度偏差测试:")
judge_len = LLMJudge(bias_length=0.2)
response_short = "梯度下降是优化算法。"
response_long = "梯度下降是优化算法。" + "详细解释" * 10

winner3, sa3, sb3 = judge_len.judge(prompt, response_short, response_long)
print(f"  短 vs 长: 判 {winner3} (短={sa3:.2f}, 长={sb3:.2f})")
print("→ Judge 可能偏好长回答，即使内容相同")""")

code("""# 减少偏差的方法
def judge_with_debiasing(judge, prompt, response_a, response_b):
    # 1. 位置去偏差: 交换 A/B 再判一次
    winner1, sa1, sb1 = judge.judge(prompt, response_a, response_b)
    winner2, sa2, sb2 = judge.judge(prompt, response_b, response_a)

    # 2. 取平均
    score_a = (sa1 + sb2) / 2
    score_b = (sb1 + sa2) / 2

    return 'A' if score_a > score_b else 'B', score_a, score_b

print("去偏差后:")
winner, sa, sb = judge_with_debiasing(judge, prompt, response_good, response_bad)
print(f"  好 vs 坏: 判 {winner} (好={sa:.2f}, 坏={sb:.2f})")
print("\\n去偏差方法: 交换位置取平均 → 减少位置偏差。")""")

md("""## 5. 生产评估

### 5.1 离线 vs 在线评估

```
离线评估 (Offline):
  - 在固定 benchmark 上测试
  - 发布前
  - 快速、可控
  - 可能不反映真实使用

在线评估 (Online):
  - A/B 测试
  - 真实用户
  - 发布后
  - 真实但慢、有风险

→ 两者都需要: 离线筛选 + 在线验证
```

### 5.2 在线指标

```
1. 用户满意度
   - 点赞/踩
   - 重新生成率 (不满意 → 重生成)
   - 停留时间

2. 任务完成率
   - 用户是否完成目标任务
   - 后续是否需要人工修正

3. 安全指标
   - 有害输出率
   - 越狱成功率
   - 投诉率

4. 延迟/成本
   - 首 token 延迟
   - 吞吐量
   - 每请求成本
```

> 💡 生产评估不只看"质量"——安全、延迟、成本同样重要。""")

code("""# A/B 测试模拟
np.random.seed(42)

n_users = 1000

# 模型 A (当前) 和 模型 B (新)
# 用户满意度 (1-5 星)
satisfaction_A = np.random.normal(3.8, 1.0, n_users).clip(1, 5)
satisfaction_B = np.random.normal(4.1, 0.9, n_users).clip(1, 5)

# 重新生成率 (越低越好)
regen_A = np.random.binomial(1, 0.25, n_users)
regen_B = np.random.binomial(1, 0.18, n_users)

# 延迟 (秒)
latency_A = np.random.exponential(2.0, n_users)
latency_B = np.random.exponential(2.5, n_users)  # B 更慢

# 有害输出率
harm_A = np.random.binomial(1, 0.02, n_users)
harm_B = np.random.binomial(1, 0.01, n_users)

print("A/B 测试结果:")
print("=" * 50)
print(f"{'指标':>20} | {'模型 A':>10} | {'模型 B':>10} | {'差异':>10}")
print("-" * 55)
print(f"{'满意度 (1-5)':>20} | {satisfaction_A.mean():>10.3f} | {satisfaction_B.mean():>10.3f} | {satisfaction_B.mean()-satisfaction_A.mean():>+10.3f}")
print(f"{'重生成率':>20} | {regen_A.mean():>10.3f} | {regen_B.mean():>10.3f} | {regen_B.mean()-regen_A.mean():>+10.3f}")
print(f"{'延迟 (秒)':>20} | {latency_A.mean():>10.3f} | {latency_B.mean():>10.3f} | {latency_B.mean()-latency_A.mean():>+10.3f}")
print(f"{'有害率':>20} | {harm_A.mean():>10.3f} | {harm_B.mean():>10.3f} | {harm_B.mean()-harm_A.mean():>+10.3f}")

print("\\n决策: B 满意度更高、更安全，但延迟更高 → 需要权衡。")""")

code("""# A/B 测试可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 满意度分布
ax = axes[0, 0]
ax.hist(satisfaction_A, bins=20, alpha=0.6, color='blue', label='模型 A', density=True)
ax.hist(satisfaction_B, bins=20, alpha=0.6, color='red', label='模型 B', density=True)
ax.set_xlabel('满意度 (1-5)', fontsize=11)
ax.set_title('用户满意度分布', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

# 重生成率
ax = axes[0, 1]
models = ['模型 A', '模型 B']
regen_rates = [regen_A.mean(), regen_B.mean()]
ax.bar(models, regen_rates, color=['blue', 'red'], alpha=0.7)
ax.set_ylabel('重生成率', fontsize=11)
ax.set_title('重生成率 (越低越好)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 延迟分布
ax = axes[1, 0]
ax.hist(latency_A, bins=30, alpha=0.6, color='blue', label='模型 A', density=True)
ax.hist(latency_B, bins=30, alpha=0.6, color='red', label='模型 B', density=True)
ax.set_xlabel('延迟 (秒)', fontsize=11)
ax.set_title('延迟分布', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

# 有害率
ax = axes[1, 1]
harm_rates = [harm_A.mean(), harm_B.mean()]
ax.bar(models, harm_rates, color=['blue', 'red'], alpha=0.7)
ax.set_ylabel('有害输出率', fontsize=11)
ax.set_title('有害输出率 (越低越好)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_ab_test.png', bbox_inches='tight')
plt.show()
print("A/B 测试: 多维度对比 → 综合决策是否上线新模型。")""")

md("""## 6. 评估的层级

### 6.1 多层评估

```
Level 0: 自动化测试
  - 语法正确性
  - 不崩溃
  - 延迟/成本

Level 1: Benchmark 评估
  - MMLU/HumanEval/GSM8K
  - 离线、快速

Level 2: LLM-as-Judge
  - 开放式问答
  - 成本低

Level 3: 人工评估
  - 专家评审
  - 高质量、高成本

Level 4: 在线 A/B
  - 真实用户
  - 最终验证

→ 每层过滤一部分 → 逐层精筛
```

> 💡 评估是漏斗——从自动化到在线 A/B，每层筛选，逐层精筛。""")

code("""# 评估漏斗
fig, ax = plt.subplots(figsize=(10, 6))

levels = ['Level 0\\n自动化', 'Level 1\\nBenchmark', 'Level 2\\nLLM-Judge', 'Level 3\\n人工', 'Level 4\\nA/B']
candidates = [1000, 500, 200, 50, 5]
colors = ['steelblue', 'forestgreen', 'coral', 'purple', 'crimson']

bars = ax.barh(levels, candidates, color=colors, alpha=0.8, edgecolor='black')
ax.set_xlabel('候选模型数量', fontsize=12)
ax.set_title('评估漏斗：从 1000 个到 5 个', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for bar, n in zip(bars, candidates):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f'{n}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('notebooks/fig_eval_funnel.png', bbox_inches='tight')
plt.show()
print("评估漏斗: 每层过滤一部分候选 → 最终选出最佳模型。")""")

md("""## 7. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 评估的挑战 | ✅ |
| 主要 LLM Benchmark | ✅ |
| Benchmark 设计原则 | ✅ |
| 数据污染与动态 benchmark | ✅ |
| LLM-as-Judge 及其偏差 | ✅ |
| 生产评估 (A/B 测试) | ✅ |
| 评估漏斗 | ✅ |

### 核心 takeaway
> **评估是漏斗**——从自动化测试到在线 A/B，逐层精筛。LLM-as-Judge 降低成本但有偏差，动态 benchmark 防污染。没有完美的评估，只有多层评估的组合。

### 🔗 下一板块
**`52_agent_foundations.ipynb`** — ReAct、Plan-Execute、ToT/GoT（进入板块九：Agent 与系统）

---

> 💬 **板块八(对齐、安全与评估)100%完成 (5/5)。**""")

output_path = "notebooks/51_evaluation.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")