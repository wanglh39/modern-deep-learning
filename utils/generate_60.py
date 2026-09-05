# 生成 60_agent_evaluation.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 60 — Agent 评估与 Benchmark

> Agent 比传统 LLM 更难评估——不只看输出质量，还要看任务完成。

## 本章你将掌握

1. **Agent 评估挑战**：为什么更难
2. **评估维度**：成功率/效率/成本/安全
3. **主要 Benchmark**：AgentBench/SWE-bench/WebArena
4. **评估方法**：端到端/分步/人工""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
print("环境准备完成 ✅")""")

md("""## 1. Agent 评估的挑战

### 1.1 为什么更难？

```
LLM 评估:
  - 输入 → 输出 → 评分
  - 单轮，静态

Agent 评估:
  - 多步，动态
  - 涉及工具调用
  - 环境状态变化
  - 任务完成度难定义

挑战:
  1. 多步性: 一步错 → 全盘错？
  2. 环境依赖: 同一 Agent 不同环境表现不同
  3. 工具调用: 调用对了吗？顺序对吗？
  4. 完成度: 部分完成算不算？
  5. 效率: 走了 10 步 vs 3 步
```""")

md("""## 2. 评估维度

### 2.1 多维度评估

```
1. 成功率 (Success Rate)
   - 任务是否完成
   - 最核心指标

2. 效率 (Efficiency)
   - 步数: 多少步完成
   - 时间: 多久完成
   - 工具调用次数

3. 成本 (Cost)
   - token 消耗
   - API 调用费用
   - 计算资源

4. 安全 (Safety)
   - 有害操作率
   - 越权操作
   - 数据泄露

5. 鲁棒性 (Robustness)
   - 不同环境表现一致
   - 输入扰动不崩溃
```""")

code("""# Agent 评估框架
class AgentEvaluator:
    def __init__(self):
        self.metrics = defaultdict(list)

    def evaluate_run(self, task, agent_run):
        # 1. 成功率
        success = agent_run.get('success', False)
        self.metrics['success_rate'].append(1 if success else 0)

        # 2. 效率
        steps = agent_run.get('steps', 0)
        self.metrics['steps'].append(steps)
        time = agent_run.get('time', 0)
        self.metrics['time'].append(time)

        # 3. 成本
        tokens = agent_run.get('tokens', 0)
        self.metrics['tokens'].append(tokens)
        cost = agent_run.get('cost', 0)
        self.metrics['cost'].append(cost)

        # 4. 安全
        harmful = agent_run.get('harmful_actions', 0)
        self.metrics['harmful'].append(harmful)

    def summary(self):
        result = {}
        result['成功率'] = np.mean(self.metrics['success_rate'])
        result['平均步数'] = np.mean(self.metrics['steps'])
        result['平均时间'] = np.mean(self.metrics['time'])
        result['平均token'] = np.mean(self.metrics['tokens'])
        result['平均成本'] = np.mean(self.metrics['cost'])
        result['有害操作'] = np.mean(self.metrics['harmful'])
        return result

# 模拟多个 Agent 运行
evaluator = AgentEvaluator()

np.random.seed(42)
for _ in range(20):
    success = np.random.random() > 0.4
    evaluator.evaluate_run("task", {
        'success': success,
        'steps': np.random.randint(3, 15),
        'time': np.random.uniform(5, 30),
        'tokens': np.random.randint(1000, 5000),
        'cost': np.random.uniform(0.01, 0.1),
        'harmful_actions': 1 if np.random.random() < 0.05 else 0,
    })

summary = evaluator.summary()
print("Agent 评估总结:")
print("=" * 40)
for metric, value in summary.items():
    if '率' in metric or '操作' in metric:
        print(f"  {metric}: {value:.2%}")
    else:
        print(f"  {metric}: {value:.2f}")""")

md("""## 3. 主要 Agent Benchmark

### 3.1 Benchmark 列表

```
AgentBench:
  - 多种任务环境
  - 交互式评估
  - OS/DB/Web/Game

SWE-bench:
  - 真实 GitHub issue
  - 代码修复
  - 测试验证

WebArena:
  - 网页任务
  - 真实网站环境
  - 端到端评估

GAIA:
  - 通用 AI 助手评估
  - 多步骤推理
  - 工具使用

τ-bench:
  - 工具使用评估
  - 多轮对话
  - API 调用
```""")

code("""# Agent Benchmark 对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 各 benchmark 特点
ax = axes[0]
benchmarks = ['AgentBench', 'SWE-bench', 'WebArena', 'GAIA', 'τ-bench']
realism = [7, 9, 8, 7, 6]
difficulty = [7, 9, 6, 8, 7]

x = np.arange(len(benchmarks))
width = 0.35
ax.bar(x - width/2, realism, width, label='真实性', color='steelblue', alpha=0.8)
ax.bar(x + width/2, difficulty, width, label='难度', color='coral', alpha=0.8)
ax.set_xlabel('Benchmark', fontsize=11)
ax.set_ylabel('评分', fontsize=11)
ax.set_title('Agent Benchmark 特点', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=9, rotation=15)
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)

# 各模型在 AgentBench 上的表现
ax = axes[1]
models = ['GPT-3.5', 'GPT-4', 'Claude-3', 'GPT-4o', 'Claude-3.5']
scores = [25, 75, 80, 85, 90]
colors = ['gray', 'steelblue', 'coral', 'forestgreen', 'purple']
ax.bar(models, scores, color=colors, alpha=0.8)
ax.set_ylabel('AgentBench 分数', fontsize=11)
ax.set_title('模型在 AgentBench 上的表现', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('notebooks/fig_agent_benchmarks.png', bbox_inches='tight')
plt.show()
print("SWE-bench 最真实最难; 更强模型 Agent 能力更强。")""")

md("""## 4. 评估方法

### 4.1 方法分类

```
1. 端到端评估
   - 只看最终结果
   - 任务完成 or not
   - 简单但可能遗漏细节

2. 分步评估
   - 每步都评分
   - 可以定位问题
   - 但需要更多标注

3. 轨迹评估
   - 评估整个执行轨迹
   - 步骤是否合理
   - 是否有更优路径

4. 人工评估
   - 专家评审
   - 高质量但昂贵
   - 金标准

5. LLM-as-Judge
   - 用 LLM 评估 Agent
   - 成本低但有偏差
   - 适合大规模
```""")

code("""# 分步评估示例
class StepByStepEvaluator:
    def __init__(self):
        self.step_scores = []

    def evaluate_step(self, step_num, action, expected, actual):
        # 评估单步
        if expected in actual or actual in expected:
            score = 1.0
        else:
            score = 0.0
        self.step_scores.append((step_num, action, score))
        return score

    def evaluate_trajectory(self, trajectory):
        # 评估整个轨迹
        total = 0
        for step in trajectory:
            score = self.evaluate_step(*step)
            total += score
        return total / len(trajectory)

# 模拟 Agent 执行轨迹
trajectory = [
    (1, "search", "搜索天气", "搜索天气信息"),  # 正确
    (2, "read", "读取结果", "读取天气结果"),    # 正确
    (3, "calculate", "计算温差", "错误操作"),    # 错误
    (4, "answer", "给出答案", "给出温差答案"),   # 正确
]

evaluator = StepByStepEvaluator()
overall = evaluator.evaluate_trajectory(trajectory)

print("分步评估:")
print("=" * 50)
for step_num, action, score in evaluator.step_scores:
    status = "✅" if score > 0 else "❌"
    print(f"  步骤 {step_num} [{action}]: {status} (分数: {score:.1f})")
print(f"\\n总体得分: {overall:.2%}")""")

md("""## 5. Agent 评估最佳实践

### 5.1 建议

```
1. 多维度评估
   - 不只看成功率
   - 还看效率、成本、安全

2. 多环境测试
   - 不同任务环境
   - 避免过拟合到单一环境

3. 对比基线
   - 与简单方法对比
   - 与人类对比
   - 确认 Agent 真的有用

4. 统计显著性
   - 多次运行取平均
   - 报告方差
   - 避免 cherry-pick

5. 持续评估
   - 模型更新后重评估
   - 监控性能退化
   - 回归测试
```""")

code("""# 评估最佳实践可视化
fig, ax = plt.subplots(figsize=(10, 6))

practices = ['多维度', '多环境', '对比基线', '统计显著', '持续评估']
importance = [9, 8, 7, 8, 9]
difficulty = [5, 7, 4, 6, 8]

x = np.arange(len(practices))
width = 0.35

ax.bar(x - width/2, importance, width, label='重要性', color='steelblue', alpha=0.8)
ax.bar(x + width/2, difficulty, width, label='实现难度', color='coral', alpha=0.8)

ax.set_xlabel('最佳实践', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('Agent 评估最佳实践', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(practices, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_agent_eval_practices.png', bbox_inches='tight')
plt.show()
print("多维度和持续评估最重要; 持续评估最难实现。")""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| Agent 评估挑战 | ✅ |
| 多维度评估 | ✅ |
| 主要 Benchmark | ✅ |
| 评估方法 (端到端/分步/轨迹) | ✅ |
| 最佳实践 | ✅ |

### 核心 takeaway
> **Agent 评估是多维度的**——成功率、效率、成本、安全缺一不可。SWE-bench/WebArena 是主要 benchmark，分步评估能定位问题。

### 🔗 下一板块
**`61_time_series_models.ipynb`** — PatchTST/iTransformer 时序模型（进入板块十：时序深度学习）

---

> 💬 **板块九(Agent 与系统)100%完成 (9/9)。**""")

output_path = "notebooks/60_agent_evaluation.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")