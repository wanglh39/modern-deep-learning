# 生成 57_multi_agent_frameworks.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 57 — 多 Agent 框架：LangGraph、AutoGen、CrewAI

> 🐝 一个 Agent 能力有限，多个 Agent 协作才能完成复杂任务。

## 本章你将掌握

1. **为什么需要多 Agent**：分工协作
2. **LangGraph**：图结构编排
3. **AutoGen**：对话式协作
4. **CrewAI**：角色分工
5. **协作模式**：串行/并行/层次""")

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

md("""## 1. 为什么需要多 Agent？

### 1.1 单 Agent 的局限

```
单 Agent 问题:
  - 一个 prompt 做所有事 → 不专注
  - context 被各种信息占满 → 效率低
  - 一个 Agent 出错 → 整个任务失败

多 Agent 优势:
  - 分工: 每个 Agent 专注一件事
  - 并行: 多个 Agent 同时工作
  - 容错: 一个出错其他可以补救
  - 专业: 不同 Agent 用不同模型/prompt
```

### 1.2 协作模式

```
1. 串行 (Pipeline)
   A → B → C → 结果
   每步一个 Agent

2. 并行 (Parallel)
   A → 结果
   B → 结果 → 合并
   C → 结果

3. 层次 (Hierarchical)
   Manager
   ├── Worker A
   ├── Worker B
   └── Worker C

4. 对话 (Conversation)
   A ↔ B ↔ C
   Agent 间对话讨论

5. 竞争 (Debate)
   A vs B → Judge
   多个方案 → 选最佳
```

> 💡 多 Agent = 分工协作——就像团队，每个人专注自己的领域。""")

md("""## 2. LangGraph：图结构编排

### 2.1 核心思想

```
LangGraph:
  把 Agent 工作流建模为图
  - 节点: Agent 或操作
  - 边: 数据流 / 控制流
  - 条件边: 根据状态决定下一步

特点:
  - 灵活的拓扑 (任意图)
  - 支持循环 (迭代改进)
  - 状态管理 (共享状态)
  - 可视化 (看到工作流)
```

### 2.2 实现""")

code("""# LangGraph 简化实现
class LangGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.state = {}

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, src, dst):
        if src not in self.edges:
            self.edges[src] = []
        self.edges[src].append(('normal', dst))

    def add_conditional_edge(self, src, condition, mapping):
        if src not in self.edges:
            self.edges[src] = []
        for key, dst in mapping.items():
            self.edges[src].append(('cond', condition, key, dst))

    def run(self, entry, max_steps=20):
        current = entry
        steps = []

        for i in range(max_steps):
            if current not in self.nodes:
                break

            # 执行节点
            result = self.nodes[current](self.state)
            self.state[current] = result
            steps.append((current, result))

            # 决定下一步
            if current not in self.edges:
                break

            next_node = None
            for edge in self.edges[current]:
                if edge[0] == 'normal':
                    next_node = edge[1]
                    break
                elif edge[0] == 'cond':
                    _, condition, key, dst = edge
                    if condition(self.state) == key:
                        next_node = dst
                        break

            if next_node is None:
                break
            current = next_node

        return steps

# 构建工作流
graph = LangGraph()

def research_agent(state):
    return "研究了主题: AI"

def write_agent(state):
    return f"写了文章关于: {state.get('research', '')}"

def review_agent(state):
    return "审查通过" if "AI" in state.get('write', '') else "需要修改"

def publish_agent(state):
    return f"发布: {state.get('write', '')}"

graph.add_node("research", research_agent)
graph.add_node("write", write_agent)
graph.add_node("review", review_agent)
graph.add_node("publish", publish_agent)

graph.add_edge("research", "write")
graph.add_edge("write", "review")
graph.add_conditional_edge("review",
    lambda s: "pass" if "通过" in s.get('review', '') else "fail",
    {"pass": "publish", "fail": "write"})

steps = graph.run("research")
print("LangGraph 工作流:")
print("=" * 50)
for i, (node, result) in enumerate(steps):
    print(f"  {i+1}. {node}: {result}")""")

md("""## 3. AutoGen：对话式协作

### 3.1 核心思想

```
AutoGen:
  Agent 间通过对话协作
  - UserProxy: 代表用户
  - AssistantAgent: AI 助手
  - 多 Agent 对话: GroupChat

特点:
  - 对话式 (自然交互)
  - 支持代码执行
  - 灵活的 Agent 角色
  - GroupChat 自动路由
```""")

code("""# AutoGen 简化实现
class Agent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.messages = []

    def respond(self, message, context):
        # 模拟 Agent 回复
        if self.role == "coder":
            return f"代码: def solve(): return '{message}'"
        elif self.role == "reviewer":
            return "审查: 代码看起来正确"
        elif self.role == "user":
            return f"用户需求: {message}"
        else:
            return f"收到: {message}"

class GroupChat:
    def __init__(self, agents, max_rounds=5):
        self.agents = agents
        self.max_rounds = max_rounds
        self.history = []

    def chat(self, initial_message):
        self.history.append(("User", initial_message))

        for round_num in range(self.max_rounds):
            for agent in self.agents:
                context = "\\n".join([f"{n}: {m}" for n, m in self.history[-3:]])
                response = agent.respond(initial_message if round_num == 0 else self.history[-1][1], context)
                self.history.append((agent.name, response))

                if "完成" in response or "正确" in response:
                    return self.history

        return self.history

# 演示
coder = Agent("Coder", "coder")
reviewer = Agent("Reviewer", "reviewer")

chat = GroupChat([coder, reviewer], max_rounds=3)
history = chat.chat("写一个排序函数")

print("AutoGen GroupChat:")
print("=" * 50)
for name, msg in history:
    print(f"  {name}: {msg}")""")

md("""## 4. CrewAI：角色分工

### 4.1 核心思想

```
CrewAI:
  把 Agent 当作"团队成员"
  - Agent: 有角色、目标、背景
  - Task: 有描述、负责 Agent
  - Crew: 团队，管理 Agent + Task
  - Process: 串行 or 并行

特点:
  - 角色明确 (每个 Agent 有 role)
  - 目标驱动 (每个 Task 有 goal)
  - 简单直观 (像组建团队)
```""")

code("""# CrewAI 简化实现
class CrewAgent:
    def __init__(self, role, goal, backstory=""):
        self.role = role
        self.goal = goal
        self.backstory = backstory

    def execute(self, task):
        return f"{self.role} 完成: {task}"

class Task:
    def __init__(self, description, agent, expected_output=""):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output

class Crew:
    def __init__(self, agents, tasks, process="sequential"):
        self.agents = agents
        self.tasks = tasks
        self.process = process

    def run(self):
        results = []
        if self.process == "sequential":
            for task in self.tasks:
                result = task.agent.execute(task.description)
                results.append(result)
        elif self.process == "parallel":
            for task in self.tasks:
                result = task.agent.execute(task.description)
                results.append(result)
        return results

# 组建团队
researcher = CrewAgent("研究员", "收集信息", "擅长搜索和分析")
writer = CrewAgent("作家", "写文章", "擅长写作")
editor = CrewAgent("编辑", "审查和修改", "擅长编辑")

# 分配任务
tasks = [
    Task("研究AI最新进展", researcher),
    Task("写一篇科普文章", writer),
    Task("审查文章质量", editor),
]

# 组建 Crew
crew = Crew([researcher, writer, editor], tasks, process="sequential")
results = crew.run()

print("CrewAI 团队:")
print("=" * 50)
for i, result in enumerate(results):
    print(f"  {i+1}. {result}")""")

md("""## 5. 框架对比

### 5.1 总结

```
框架       风格        优点              缺点
LangGraph  图结构      灵活、可视化      学习曲线陡
AutoGen    对话式      自然、支持代码    可能发散
CrewAI     角色分工    简单直观          灵活性不足
```""")

code("""# 框架对比
fig, ax = plt.subplots(figsize=(10, 6))

frameworks = ['LangGraph', 'AutoGen', 'CrewAI']
flexibility = [9, 7, 5]
ease_of_use = [4, 6, 9]
capability = [9, 8, 7]

x = np.arange(len(frameworks))
width = 0.25

ax.bar(x - width, flexibility, width, label='灵活性', color='steelblue', alpha=0.8)
ax.bar(x, ease_of_use, width, label='易用性', color='coral', alpha=0.8)
ax.bar(x + width, capability, width, label='能力', color='forestgreen', alpha=0.8)

ax.set_xlabel('框架', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('多 Agent 框架对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(frameworks, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_multi_agent.png', bbox_inches='tight')
plt.show()
print("LangGraph 最灵活; CrewAI 最易用; AutoGen 居中。")""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 多 Agent 协作模式 | ✅ |
| LangGraph (图结构) | ✅ |
| AutoGen (对话式) | ✅ |
| CrewAI (角色分工) | ✅ |
| 框架对比 | ✅ |

### 核心 takeaway
> **多 Agent = 分工协作**——LangGraph 用图编排，AutoGen 用对话协作，CrewAI 用角色分工。选择取决于任务复杂度和团队偏好。

### 🔗 下一章
**`58_coding_agents.ipynb`** — 代码 Agent、SWE-bench

---

> 💬 **板块九(Agent 与系统)进行中 (6/9)。**""")

output_path = "notebooks/57_multi_agent_frameworks.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")