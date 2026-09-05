# 生成 58_coding_agents.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "FSDP"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 58 — 代码 Agent 与 SWE-bench

> 🔥 从写代码到修 bug 到审 PR——代码 Agent 正在重塑软件开发。

## 本章你将掌握

1. **代码 Agent**：从补全到自主编程
2. **SWE-bench**：软件工程评估
3. **代码 Agent 架构**：规划+编辑+测试
4. **实际应用**：7B vs 70B vs 405B""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
print("环境准备完成 ✅")""")

md("""## 1. 代码 Agent 的演进

### 1.1 从补全到自主

```
演进:
  2021  Copilot    → 代码补全 (单行/多行)
  2023  ChatGPT    → 代码对话 (问答)
  2024  Cursor     → 代码编辑 (项目级)
  2024  Devin      → 自主编程 (端到端)
  2025  SWE-agent  → 修 bug (SWE-bench)

能力演进:
  补全 → 对话 → 编辑 → 规划 → 自主
```

### 1.2 代码 Agent 的能力

```
1. 代码生成
   - 从需求到代码
   - 函数/类/模块

2. Bug 修复
   - 理解错误
   - 定位问题
   - 修复+测试

3. 代码审查
   - 找潜在问题
   - 建议改进
   - 安全检查

4. 重构
   - 改善代码结构
   - 保持行为不变

5. 测试生成
   - 写单元测试
   - 边界情况
   - 覆盖率
```

> 💡 代码 Agent 从"补全工具"变成"开发伙伴"——能理解项目、规划任务、自主执行。""")

md("""## 2. SWE-bench

### 2.1 什么是 SWE-bench？

```
SWE-bench (Software Engineering Benchmark):
  - 真实 GitHub issue
  - 任务: 修复 issue (写代码 + 通过测试)
  - 评估: 测试是否通过

特点:
  - 真实: 来自真实项目 (Django, Flask, ...)
  - 端到端: 不只写代码，还要能跑
  - 难: 需要理解大代码库

当前水平:
  - 2024 初: ~2% 解决率
  - 2024 末: ~40%+ (SWE-agent)
  - 目标: 接近人类水平
```

### 2.2 SWE-bench 任务

```
输入:
  - 代码库 (整个 repo)
  - Issue 描述 (bug 报告)
  - 测试用例 (pass-to-fail 和 fail-to-pass)

输出:
  - 代码修改 (patch)

评估:
  - 应用 patch
  - 运行测试
  - fail-to-pass 测试是否通过
  - pass-to-fail 测试是否仍通过
```""")

code("""# SWE-bench 简化模拟
class SWEBenchTask:
    def __init__(self, repo, issue, tests_pass_to_fail, tests_fail_to_pass):
        self.repo = repo
        self.issue = issue
        self.tests_ptf = tests_pass_to_fail  # 修改后应该通过的
        self.tests_ftp = tests_fail_to_pass  # 修改后应该仍通过的

class CodingAgent:
    def __init__(self, name, capability=0.5):
        self.name = name
        self.capability = capability

    def solve(self, task):
        # 模拟 Agent 解决任务
        # 能力越高，越可能解决
        solve_prob = self.capability * np.random.uniform(0.5, 1.5)

        if solve_prob > 0.7:
            return True, "修复成功，测试通过"
        else:
            return False, "未能修复或测试失败"

# 创建任务
tasks = [
    SWEBenchTask("django/django", "ORM 查询 bug", ["test_query"], ["test_other"]),
    SWEBenchTask("flask/flask", "路由问题", ["test_route"], ["test_url"]),
    SWEBenchTask("scikit-learn", "分类器精度", ["test_accuracy"], ["test_fit"]),
    SWEBenchTask("pytest/pytest", "fixture 泄漏", ["test_fixture"], ["test_session"]),
    SWEBenchTask("requests/requests", "SSL 验证", ["test_ssl"], ["test_http"]),
]

# 不同能力的 Agent
agents = [
    CodingAgent("7B 模型", 0.3),
    CodingAgent("70B 模型", 0.5),
    CodingAgent("405B 模型", 0.7),
    CodingAgent("SWE-agent", 0.8),
]

print("SWE-bench 评估:")
print("=" * 60)
print(f"{'Agent':>15} | {'解决率':>8} | {'详情':>30}")
print("-" * 60)

for agent in agents:
    solved = 0
    for task in tasks:
        success, _ = agent.solve(task)
        if success:
            solved += 1
    rate = solved / len(tasks)
    print(f"{agent.name:>15} | {rate:>8.0%} | {solved}/{len(tasks)} 任务解决")""")

md("""## 3. 代码 Agent 架构

### 3.1 典型架构

```
代码 Agent 架构:
  1. 理解 (Understand)
     - 读代码库
     - 理解 issue
     - 定位相关文件

  2. 规划 (Plan)
     - 分析问题
     - 制定修改方案
     - 确定步骤

  3. 编辑 (Edit)
     - 修改代码
     - 工具: file_edit, search, view

  4. 测试 (Test)
     - 运行测试
     - 检查结果
     - 如果失败 → 回到规划

  5. 验证 (Verify)
     - 确认修复
     - 检查副作用
     - 生成 patch
```

### 3.2 关键工具

```
代码 Agent 的工具:
  - view_file: 查看文件内容
  - search_code: 搜索代码
  - edit_file: 编辑文件
  - run_test: 运行测试
  - run_command: 执行命令
  - create_file: 创建文件

→ 类似人类开发者的工作流
```""")

code("""# 代码 Agent 实现
class CodeAgent:
    def __init__(self, name):
        self.name = name
        self.actions = []

    def understand(self, repo, issue):
        self.actions.append(("understand", f"阅读 {repo}, 理解 issue: {issue}"))
        return f"理解了 {issue}"

    def plan(self, problem):
        plan = f"修改方案: 1.定位 2.修改 3.测试"
        self.actions.append(("plan", plan))
        return plan

    def edit(self, file, changes):
        self.actions.append(("edit", f"编辑 {file}: {changes}"))
        return f"已修改 {file}"

    def test(self, test_name):
        passed = np.random.random() > 0.3
        result = "通过" if passed else "失败"
        self.actions.append(("test", f"运行 {test_name}: {result}"))
        return passed

    def solve(self, repo, issue):
        # 1. 理解
        self.understand(repo, issue)

        # 2. 规划
        self.plan(issue)

        # 3. 编辑
        self.edit("src/main.py", "修复bug")

        # 4. 测试
        for attempt in range(3):
            if self.test("test_bug_fix"):
                self.actions.append(("done", "修复成功!"))
                return True
            else:
                self.actions.append(("retry", f"重试 {attempt+1}"))
                self.edit("src/main.py", "调整修复")

        self.actions.append(("fail", "未能修复"))
        return False

# 演示
agent = CodeAgent("SWE-agent")
success = agent.solve("django/django", "ORM 查询返回错误结果")

print(f"代码 Agent: {agent.name}")
print("=" * 50)
for i, (action, detail) in enumerate(agent.actions):
    print(f"  {i+1}. [{action}] {detail}")""")

md("""## 4. 代码 Agent 的挑战

### 4.1 主要挑战

```
1. 上下文长度
   - 代码库可能很大
   - 需要检索相关部分
   - 类似 RAG

2. 理解深度
   - 不只是语法，还有语义
   - 跨文件依赖
   - 设计模式

3. 测试可靠性
   - 测试通过 ≠ 修复正确
   - 可能有副作用
   - 边界情况

4. 工具使用
   - 何时搜索、何时编辑
   - 搜索什么、编辑什么
   - 工具编排策略

5. 长程规划
   - 复杂 bug 需要多步
   - 中间可能走错
   - 需要回溯
```""")

code("""# 代码 Agent 能力对比
fig, ax = plt.subplots(figsize=(12, 6))

tasks = ['代码补全', 'Bug修复', '代码审查', '重构', '端到端开发', 'SWE-bench']
models = {
    '7B': [0.8, 0.2, 0.5, 0.3, 0.1, 0.05],
    '70B': [0.9, 0.5, 0.7, 0.6, 0.3, 0.2],
    '405B': [0.95, 0.7, 0.85, 0.75, 0.5, 0.35],
    'SWE-agent': [0.9, 0.85, 0.8, 0.7, 0.6, 0.45],
}

x = np.arange(len(tasks))
width = 0.2
colors = ['steelblue', 'forestgreen', 'coral', 'purple']

for i, (model, scores) in enumerate(models.items()):
    offset = (i - 1.5) * width
    ax.bar(x + offset, scores, width, label=model, color=colors[i], alpha=0.8)

ax.set_xlabel('任务', fontsize=12)
ax.set_ylabel('成功率', fontsize=12)
ax.set_title('代码 Agent 能力对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=10, rotation=15)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('notebooks/fig_coding_agents.png', bbox_inches='tight')
plt.show()
print("模型越大能力越强; SWE-agent 在 SWE-bench 上最优。")""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 代码 Agent 演进 | ✅ |
| SWE-bench 评估 | ✅ |
| 代码 Agent 架构 | ✅ |
| 关键挑战 | ✅ |
| 模型规模 vs 能力 | ✅ |

### 核心 takeaway
> **代码 Agent 从补全到自主**——SWE-bench 是真实软件工程评估，架构是理解→规划→编辑→测试→验证。模型规模和专用工具都重要。

### 🔗 下一章
**`59_computer_use.ipynb`** — Computer use/Browser use

---

> 💬 **板块九(Agent 与系统)进行中 (7/9)。**""")

output_path = "notebooks/58_coding_agents.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")