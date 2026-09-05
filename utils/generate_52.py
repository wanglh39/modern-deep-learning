# 生成 52_agent_foundations.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 52 — Agent 基础：ReAct、Plan-Execute、ToT/GoT

> 🧭🔥 LLM 不只是"回答问题"，而是"执行任务"。Agent = LLM + 工具 + 循环。

## 本章你将掌握

1. **什么是 Agent**：从聊天到行动
2. **ReAct**：推理 + 行动交替
3. **Plan-Execute**：先规划再执行
4. **Tree of Thoughts (ToT)**：搜索思维树
5. **Graph of Thoughts (GoT)**：思维图""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
torch.manual_seed(42)
print("环境准备完成 ✅")""")

md("""## 1. 什么是 Agent？

### 1.1 从 LLM 到 Agent

```
LLM: 输入文本 → 输出文本
     被动回答，不能行动

Agent: LLM + 工具 + 循环
     主动决策，调用工具，多步推理

Agent = LLM (大脑) + Tools (手) + Memory (记忆) + Loop (循环)
```

### 1.2 Agent 核心组件

```
1. 大脑 (LLM):
   - 理解任务
   - 决策下一步
   - 解析工具输出

2. 工具 (Tools):
   - 搜索、计算、代码执行
   - API 调用、数据库查询

3. 记忆 (Memory):
   - 短期: 当前对话上下文
   - 长期: 跨会话记忆

4. 循环 (Loop):
   - 观察 → 思考 → 行动 → 观察...
   - 直到任务完成
```

> 💡 Agent 的本质是"LLM 驱动的循环"——观察、思考、行动，直到完成。""")

md("""## 2. ReAct：推理 + 行动

### 2.1 核心思想

```
ReAct (Reasoning + Acting):
  交替进行推理和行动

循环:
  Thought: 我需要搜索...
  Action: search("query")
  Observation: 搜索结果是...
  Thought: 根据结果，我需要...
  Action: calculate(...)
  Observation: 计算结果是...
  Thought: 任务完成
  Answer: 最终答案

vs 纯推理 (CoT):
  CoT: 只思考，不能获取新信息
  ReAct: 思考 + 行动，可以调用工具获取信息
```

### 2.2 实现""")

code("""# ReAct Agent 实现
class ReActAgent:
    def __init__(self, tools, max_steps=10):
        self.tools = tools
        self.max_steps = max_steps
        self.history = []

    def run(self, task):
        self.history.append(f"Task: {task}")

        for step in range(self.max_steps):
            # 模拟 LLM 思考
            thought, action, action_input = self._think(task)

            self.history.append(f"Thought {step+1}: {thought}")
            self.history.append(f"Action {step+1}: {action}({action_input})")

            if action == "Finish":
                self.history.append(f"Answer: {action_input}")
                return action_input

            # 执行工具
            if action in self.tools:
                observation = self.tools[action](action_input)
            else:
                observation = f"未知工具: {action}"

            self.history.append(f"Observation {step+1}: {observation}")

        return "达到最大步数，未能完成"

    def _think(self, task):
        # 模拟 LLM 的推理 (用简单规则)
        if "天气" in task and len(self.history) < 5:
            return "我需要搜索天气信息", "search", "天气"
        elif "搜索" in str(self.history[-1]):
            return "搜索完成，我需要计算温度差", "calculate", "25-15"
        elif "计算" in str(self.history[-1]):
            return "计算完成，温差是10度", "Finish", "温差是10度"
        else:
            return "任务完成", "Finish", "已完成"

# 定义工具
def search(query):
    return f"搜索 '{query}': 今天25度"

def calculate(expr):
    try:
        result = eval(expr)
        return f"计算 {expr} = {result}"
    except:
        return "计算失败"

tools = {"search": search, "calculate": calculate}

agent = ReActAgent(tools)
result = agent.run("今天天气怎么样？和昨天比差多少？")

print("ReAct Agent 执行轨迹:")
print("=" * 50)
for line in agent.history:
    print(line)""")

md("""## 3. Plan-Execute：先规划再执行

### 3.1 核心思想

```
Plan-Execute:
  Step 1: 规划 (Plan)
    把大任务分解为子步骤列表

  Step 2: 执行 (Execute)
    依次执行每个子步骤

  Step 3: 重规划 (Re-plan) [可选]
    如果执行中发现计划有问题 → 重新规划

vs ReAct:
  ReAct: 每步都思考 (灵活但可能发散)
  Plan-Execute: 先规划 (有方向但可能不灵活)
```

### 3.2 实现""")

code("""# Plan-Execute Agent
class PlanExecuteAgent:
    def __init__(self, tools):
        self.tools = tools

    def plan(self, task):
        # 模拟 LLM 规划
        if "写报告" in task:
            return [
                "搜索相关信息",
                "整理要点",
                "写大纲",
                "写正文",
                "检查格式",
            ]
        else:
            return ["理解任务", "执行", "验证结果"]

    def execute_step(self, step):
        # 模拟执行
        return f"✅ 完成: {step}"

    def run(self, task):
        print(f"任务: {task}")
        print("=" * 50)

        # Step 1: 规划
        plan = self.plan(task)
        print(f"\\n规划 ({len(plan)} 步):")
        for i, step in enumerate(plan):
            print(f"  {i+1}. {step}")

        # Step 2: 执行
        print(f"\\n执行:")
        results = []
        for i, step in enumerate(plan):
            result = self.execute_step(step)
            results.append(result)
            print(f"  {i+1}. {result}")

        # Step 3: 检查是否需要重规划
        if any("失败" in r for r in results):
            print("\\n⚠️ 需要重规划...")
        else:
            print("\\n✅ 所有步骤完成，任务成功！")

        return results

agent = PlanExecuteAgent(tools={})
results = agent.run("写一篇关于AI的报告")""")

md("""## 4. Tree of Thoughts (ToT)

### 4.1 核心思想

```
ToT (Tree of Thoughts):
  把推理过程组织成树
  - 每个节点是一个"思维状态"
  - 分支: 多种可能的下一步
  - 评估: 给每个状态打分
  - 搜索: BFS/DFS/Beam search

vs CoT (Chain of Thoughts):
  CoT: 单条链 (一条路走到底)
  ToT: 树搜索 (探索多条路，回溯)

适用场景:
  - 需要探索的复杂问题
  - 初始方向可能错的问题
  - 需要前瞻和回溯的问题
```

### 4.2 实现""")

code("""# Tree of Thoughts 实现
class ThoughtNode:
    def __init__(self, thought, parent=None):
        self.thought = thought
        self.parent = parent
        self.children = []
        self.score = 0.0
        self.visited = False

    def add_child(self, thought):
        child = ThoughtNode(thought, self)
        self.children.append(child)
        return child

class ToTAgent:
    def __init__(self, max_depth=3, branching=3, beam_width=2):
        self.max_depth = max_depth
        self.branching = branching
        self.beam_width = beam_width

    def generate_thoughts(self, current_thought, n):
        # 模拟 LLM 生成 n 个可能的下一步思考
        thoughts = []
        for i in range(n):
            thoughts.append(f"{current_thought} → 方案{i+1}")
        return thoughts

    def evaluate_thought(self, thought):
        # 模拟评估思维状态 (0-1 分)
        depth = thought.count("→")
        return np.random.uniform(0.3, 0.9) * (1 - depth * 0.1)

    def search(self, initial_thought):
        root = ThoughtNode(initial_thought)
        root.score = 1.0

        # Beam search
        beam = [root]

        for depth in range(self.max_depth):
            candidates = []
            for node in beam:
                thoughts = self.generate_thoughts(node.thought, self.branching)
                for t in thoughts:
                    child = node.add_child(t)
                    child.score = self.evaluate_thought(t)
                    candidates.append(child)

            # 选 top-k
            candidates.sort(key=lambda x: x.score, reverse=True)
            beam = candidates[:self.beam_width]

        # 返回最佳路径
        best = max(beam, key=lambda x: x.score)
        path = []
        node = best
        while node:
            path.append(node.thought)
            node = node.parent
        path.reverse()
        return path, best.score

agent = ToTAgent(max_depth=3, branching=3, beam_width=2)
path, score = agent.search("解决问题")

print("Tree of Thoughts 搜索结果:")
print("=" * 50)
for i, thought in enumerate(path):
    print(f"  深度 {i}: {thought} (分数: {score:.3f})")
print(f"\\n最佳路径分数: {score:.3f}")""")

code("""# ToT 可视化
fig, ax = plt.subplots(figsize=(14, 8))

# 构建树可视化
np.random.seed(42)
tree_data = {
    'Root': {
        'A': {'A1': None, 'A2': None, 'A3': None},
        'B': {'B1': None, 'B2': None, 'B3': None},
        'C': {'C1': None, 'C2': None, 'C3': None},
    }
}

# 画树
def draw_tree(ax, x, y, dx, dy, label, children, depth=0, max_depth=2):
    color = 'steelblue' if depth == 0 else ('forestgreen' if depth == 1 else 'coral')
    ax.scatter(x, y, s=200, c=color, zorder=5, edgecolors='black')
    ax.annotate(label, (x, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    if children and depth < max_depth:
        n = len(children)
        for i, (child_label, grandchildren) in enumerate(children.items()):
            child_x = x + (i - (n-1)/2) * dx
            child_y = y - dy
            ax.plot([x, child_x], [y, child_y], 'k-', alpha=0.3, zorder=1)
            draw_tree(ax, child_x, child_y, dx/2, dy, child_label, grandchildren, depth+1, max_depth)

draw_tree(ax, 0, 0, 3, 2.5, 'Root', tree_data['Root'])

# 标注最佳路径
ax.annotate('最佳路径', xy=(1.5, -2.5), fontsize=12, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.set_xlim(-5, 5); ax.set_ylim(-8, 1.5)
ax.set_title('Tree of Thoughts: 思维树搜索', fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('notebooks/fig_tot.png', bbox_inches='tight')
plt.show()
print("ToT: 从根节点展开多分支，评估后选最优路径 → 比单链CoT更强。")""")

md("""## 5. Graph of Thoughts (GoT)

### 5.1 从树到图

```
ToT 的限制:
  - 树结构: 不能合并分支
  - 同一子问题可能在不同分支重复计算

GoT (Graph of Thoughts):
  - 思维图: 节点可以有多父节点
  - 支持合并: 两个分支的结果可以合并
  - 支持回溯和循环

GoT 操作:
  1. 生成: 新思维节点
  2. 评估: 给节点打分
  3. 合并: 多个节点 → 一个
  4. 回溯: 放弃差的分支
  5. 精化: 改进已有节点
```

### 5.2 GoT vs ToT

```
          CoT          ToT          GoT
结构      链           树            图
分支      无           有            有
合并      无           无            有
回溯      无           有            有
循环      无           无            有
表达力    弱           中            强
成本      低           中            高
```

> 💡 GoT 是最通用的思维结构——支持合并、回溯、循环，但成本也最高。""")

code("""# GoT 实现 (简化)
class ThoughtGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.node_id = 0

    def add_node(self, thought, score=0.0):
        nid = self.node_id
        self.nodes[nid] = {'thought': thought, 'score': score}
        self.node_id += 1
        return nid

    def add_edge(self, src, dst):
        self.edges.append((src, dst))

    def merge(self, node_ids, new_thought):
        # 合并多个节点
        new_id = self.add_node(new_thought)
        for nid in node_ids:
            self.add_edge(nid, new_id)
        # 合并分数
        avg_score = np.mean([self.nodes[nid]['score'] for nid in node_ids])
        self.nodes[new_id]['score'] = avg_score + 0.1  # 合并奖励
        return new_id

    def best_path(self):
        best_id = max(self.nodes, key=lambda x: self.nodes[x]['score'])
        return best_id, self.nodes[best_id]

# 构建 GoT
got = ThoughtGraph()

# 分支1
a1 = got.add_node("思路A1: 分析问题", 0.6)
a2 = got.add_node("思路A2: 搜索资料", 0.7)
got.add_edge(a1, a2)

# 分支2
b1 = got.add_node("思路B1: 类比推理", 0.5)
b2 = got.add_node("思路B2: 案例分析", 0.8)
got.add_edge(b1, b2)

# 合并两个分支
merged = got.merge([a2, b2], "综合两条思路的结论")

# 精化
final = got.add_node("最终答案", 0.95)
got.add_edge(merged, final)

print("Graph of Thoughts:")
print("=" * 50)
for nid, node in got.nodes.items():
    print(f"  节点 {nid}: {node['thought']} (分数: {node['score']:.2f})")

print(f"\\n边: {got.edges}")
best_id, best_node = got.best_path()
print(f"\\n最佳节点: {best_id} → {best_node['thought']} (分数: {best_node['score']:.2f})")""")

md("""## 6. Agent 范式对比

### 6.1 总结

```
范式          结构      优点              缺点
CoT          链        简单              不能探索
ReAct        链+工具   能获取信息        可能发散
Plan-Execute 链+计划   有方向            不够灵活
ToT          树        能探索回溯        成本高
GoT          图        最灵活            最复杂
```

### 6.2 选择建议

```
简单问答 → CoT
需要工具 → ReAct
复杂任务 → Plan-Execute
需要探索 → ToT
需要合并多思路 → GoT
```""")

code("""# Agent 范式对比可视化
fig, ax = plt.subplots(figsize=(12, 7))

paradigms = ['CoT', 'ReAct', 'Plan-Execute', 'ToT', 'GoT']
flexibility = [2, 4, 3, 7, 9]
cost = [1, 3, 2, 6, 9]
capability = [3, 5, 4, 7, 9]

x = np.arange(len(paradigms))
width = 0.25

ax.bar(x - width, flexibility, width, label='灵活性', color='steelblue', alpha=0.8)
ax.bar(x, cost, width, label='成本', color='coral', alpha=0.8)
ax.bar(x + width, capability, width, label='能力', color='forestgreen', alpha=0.8)

ax.set_xlabel('Agent 范式', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('Agent 范式对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(paradigms, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_agent_paradigms.png', bbox_inches='tight')
plt.show()
print("CoT 最简单; GoT 最强但最贵; ReAct 性价比最高。")""")

md("""## 7. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| Agent = LLM + 工具 + 记忆 + 循环 | ✅ |
| ReAct (推理+行动) | ✅ |
| Plan-Execute (规划+执行) | ✅ |
| Tree of Thoughts (思维树搜索) | ✅ |
| Graph of Thoughts (思维图) | ✅ |
| 范式对比与选择 | ✅ |

### 核心 takeaway
> **Agent 是 LLM 驱动的循环**——ReAct 交替推理和行动，ToT 搜索思维树，GoT 支持合并和回溯。范式从链→树→图，灵活性增加但成本也增加。

### 🔗 下一章
**`53_tool_use_mcp.ipynb`** — Function Calling + MCP 协议

---

> 💬 **板块九(Agent 与系统)进行中 (1/9)。**""")

output_path = "notebooks/52_agent_foundations.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")