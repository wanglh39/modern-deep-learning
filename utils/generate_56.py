# 生成 56_memory_systems.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 56 — 记忆系统：MemGPT 与长期记忆

> Agent 如何"记住"过去的对话？MemGPT 用操作系统的思想管理记忆。

## 本章你将掌握

1. **记忆层次**：短期/工作/长期记忆
2. **MemGPT/Letta**：OS 思想的记忆管理
3. **记忆操作**：写入、检索、遗忘
4. **记忆策略**：何时记、记什么""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
print("环境准备完成 ✅")""")

md("""## 1. Agent 记忆的层次

### 1.1 三层记忆

```
1. 短期记忆 (Short-term / Working Memory)
   = 当前 context window
   - 最近对话
   - 当前任务状态
   - 容量有限 (如 128K tokens)
   - 随会话结束消失

2. 长期记忆 (Long-term Memory)
   = 外部存储
   - 过去会话
   - 用户偏好
   - 学到的事实
   - 容量无限
   - 跨会话持久

3. 反思记忆 (Reflective Memory)
   = 从经验中提炼
   - 总结过去的对话
   - 提取规律和偏好
   - 更高层的知识
```

### 1.2 记忆 vs RAG

```
RAG: 检索外部文档 (被动知识)
记忆: 检索过去经验 (主动历史)

区别:
  RAG: "资料库" → 检索事实
  记忆: "日记本" → 检索经历

联系:
  记忆系统常用 RAG 技术检索
  但存储的是 Agent 的经历，不是外部文档
```

> 💡 记忆让 Agent 有"连续性"——不只是活在当前 context，而是有过去的经验。""")

md("""## 2. MemGPT / Letta

### 2.1 核心思想

```
MemGPT (Memory-GPT) / Letta:
  用操作系统的思想管理 LLM 记忆

类比:
  OS: 管理有限 RAM + 无限磁盘
  MemGPT: 管理有限 context + 无限外部存储

  RAM = context window (有限)
  磁盘 = 外部记忆库 (无限)
  页面调度 = 记忆换入/换出

关键: LLM 自己管理记忆
  - 通过函数调用读写记忆
  - 决定什么放入 context
  - 决定什么存到外部
```

### 2.2 MemGPT 架构

```
MemGPT 架构:
  ┌──────────────────────┐
  │ Context Window (RAM) │
  │  - System prompt     │
  │  - Working memory    │
  │  - Recent messages   │
  ├──────────────────────┤
  │ Memory Functions     │
  │  - recall_memory     │ ← 检索长期记忆
  │  - archival_insert   │ ← 写入长期记忆
  │  - send_message      │
  └──────────────────────┘
          ↕
  ┌──────────────────────┐
  │ External Memory (磁盘)│
  │  - All messages      │
  │  - Summaries         │
  │  - User preferences  │
  └──────────────────────┘
```

> 💡 MemGPT = LLM 版操作系统——LLM 自己决定记忆的换入换出。""")

code("""# MemGPT 简化实现
class MemGPT:
    def __init__(self, context_limit=500):
        self.context_limit = context_limit

        # Context Window (RAM)
        self.system_prompt = "你是一个有记忆的助手。"
        self.working_memory = []  # 工作记忆
        self.recent_messages = deque(maxlen=5)  # 最近消息

        # External Memory (磁盘)
        self.archival_memory = []  # 长期记忆
        self.all_messages = []  # 所有消息

        # 记忆函数
        self.functions = {
            'recall_memory': self._recall,
            'archival_insert': self._archival_insert,
            'send_message': self._send,
        }

    def _recall(self, query):
        # 检索长期记忆
        results = [m for m in self.archival_memory if query.lower() in m.lower()]
        return results[:3] if results else ["未找到相关记忆"]

    def _archival_insert(self, content):
        self.archival_memory.append(content)
        return f"已存入长期记忆 ({len(self.archival_memory)} 条)"

    def _send(self, message):
        self.recent_messages.append(message)
        self.all_messages.append(message)
        return message

    def estimate_context(self):
        total = len(self.system_prompt)
        total += sum(len(m) for m in self.working_memory)
        total += sum(len(m) for m in self.recent_messages)
        return total

    def manage_memory(self):
        # 如果 context 接近上限，把旧的移到外部
        while self.estimate_context() > self.context_limit and self.recent_messages:
            old = self.recent_messages.popleft()
            self._archival_insert(f"[旧消息] {old}")
            return f"已将旧消息移至长期记忆"
        return "无需管理"

    def chat(self, user_input):
        # 1. 添加用户消息
        self._send(f"User: {user_input}")

        # 2. 模拟 LLM 处理 (可能调用记忆函数)
        response = f"收到: {user_input}"

        # 如果用户提到偏好，存入长期记忆
        if "喜欢" in user_input or "偏好" in user_input:
            self._archival_insert(f"用户偏好: {user_input}")
            response += " (已记住你的偏好)"

        # 如果用户问"还记得"，检索长期记忆
        if "记得" in user_input:
            recalled = self._recall(user_input)
            response += f"\\n回忆: {recalled}"

        # 3. 添加助手回复
        self._send(f"Assistant: {response}")

        # 4. 管理记忆
        self.manage_memory()

        return response

# 演示
agent = MemGPT(context_limit=300)

print("MemGPT 对话:")
print("=" * 50)

conversations = [
    "你好！",
    "我喜欢Python编程",
    "我偏好简洁的代码",
    "你还记得我的偏好吗？",
]

for msg in conversations:
    print(f"\\nUser: {msg}")
    response = agent.chat(msg)
    print(f"Assistant: {response}")

print(f"\\n长期记忆: {agent.archival_memory}")
print(f"所有消息: {len(agent.all_messages)} 条")""")

md("""## 3. 记忆操作

### 3.1 写入策略

```
何时写入长期记忆?
  1. 用户明确告知偏好/事实
  2. 重要决策点
  3. 任务完成时的总结
  4. 定期反思

写什么?
  - 事实: "用户是程序员"
  - 偏好: "用户喜欢简洁代码"
  - 事件: "2024-01-15 讨论了X"
  - 反思: "用户经常问Y，应该准备Z"
```

### 3.2 检索策略

```
何时检索?
  1. 用户问"还记得..."
  2. 当前问题与过去相关
  3. 需要用户偏好来个性化
  4. 定期回顾

检索方法:
  - 关键词匹配 (简单)
  - 语义检索 (embedding)
  - 时间衰减 (近期优先)
  - 重要性加权
```""")

code("""# 记忆系统实现
class MemorySystem:
    def __init__(self):
        self.memories = []
        self.next_id = 0

    def add(self, content, memory_type='fact', importance=0.5):
        memory = {
            'id': self.next_id,
            'content': content,
            'type': memory_type,
            'importance': importance,
            'timestamp': self.next_id,
            'access_count': 0,
        }
        self.memories.append(memory)
        self.next_id += 1
        return memory['id']

    def retrieve(self, query, top_k=3, time_decay=0.95):
        scores = []
        for mem in self.memories:
            # 关键词匹配
            kw_score = sum(1 for w in query if w in mem['content']) / max(len(query), 1)
            # 重要性
            imp_score = mem['importance']
            # 时间衰减 (越新越高)
            time_score = time_decay ** (self.next_id - mem['timestamp'])
            # 访问频率
            access_score = 1 / (1 + mem['access_count'])

            total = 0.4 * kw_score + 0.3 * imp_score + 0.2 * time_score + 0.1 * access_score
            scores.append((mem, total))

        scores.sort(key=lambda x: -x[1])
        results = []
        for mem, score in scores[:top_k]:
            mem['access_count'] += 1
            results.append((mem, score))
        return results

    def forget(self, memory_id):
        self.memories = [m for m in self.memories if m['id'] != memory_id]

    def reflect(self):
        # 反思: 从记忆中提取模式
        types = defaultdict(list)
        for mem in self.memories:
            types[mem['type']].append(mem['content'])
        summary = {}
        for t, contents in types.items():
            summary[t] = f"{len(contents)} 条: " + "; ".join(contents[:2])
        return summary

# 演示
memory = MemorySystem()

# 写入记忆
memory.add("用户是Python程序员", 'fact', 0.8)
memory.add("用户偏好简洁代码", 'preference', 0.7)
memory.add("讨论了RAG架构", 'event', 0.5)
memory.add("用户经常问性能优化", 'pattern', 0.6)
memory.add("用户不喜欢冗长解释", 'preference', 0.7)

# 检索
print("记忆检索:")
results = memory.retrieve("用户的编程偏好", top_k=3)
for mem, score in results:
    print(f"  [{score:.4f}] {mem['type']}: {mem['content']}")

# 反思
print(f"\\n反思总结:")
summary = memory.reflect()
for t, s in summary.items():
    print(f"  {t}: {s}")""")

md("""## 4. 记忆策略对比

### 4.1 策略

```
策略              描述              优缺点
全量保存          所有消息都存      完整但可能冗余
摘要保存          定期摘要后存      省空间但丢细节
重要事件          只存重要事件      精简但可能遗漏
混合              事件+摘要+偏好    最佳但复杂
```""")

code("""# 记忆策略对比
fig, ax = plt.subplots(figsize=(10, 6))

strategies = ['全量保存', '摘要保存', '重要事件', '混合']
completeness = [9, 6, 5, 8]
efficiency = [2, 7, 9, 7]
quality = [6, 7, 6, 9]

x = np.arange(len(strategies))
width = 0.25

ax.bar(x - width, completeness, width, label='完整性', color='steelblue', alpha=0.8)
ax.bar(x, efficiency, width, label='效率', color='coral', alpha=0.8)
ax.bar(x + width, quality, width, label='质量', color='forestgreen', alpha=0.8)

ax.set_xlabel('策略', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('记忆策略对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_memory_strategies.png', bbox_inches='tight')
plt.show()
print("混合策略质量最高; 重要事件最有效率。")""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 三层记忆 (短期/长期/反思) | ✅ |
| MemGPT/Letta OS 思想 | ✅ |
| 记忆写入/检索/遗忘 | ✅ |
| 记忆策略对比 | ✅ |

### 核心 takeaway
> **MemGPT 用 OS 思想管理记忆**——有限 context 是 RAM，无限存储是磁盘，LLM 自己管理换入换出。记忆让 Agent 有连续性。

### 🔗 下一章
**`57_multi_agent_frameworks.ipynb`** — 多agent、LangGraph/AutoGen/CrewAI

---

> 💬 **板块九(Agent 与系统)进行中 (5/9)。**""")

output_path = "notebooks/56_memory_systems.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")