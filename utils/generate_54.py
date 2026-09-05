# 生成 54_context_engineering.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 54 — 上下文工程：Context Loop 与压缩

> 🔥 Context is King. 在 Agent 时代，如何管理上下文是核心工程问题。

## 本章你将掌握

1. **上下文工程**：为什么 context 是核心
2. **Context Loop**：Agent 的上下文循环
3. **Context 压缩**：在有限窗口内放更多信息
4. **Context 路由**：选择相关上下文
5. **Context 窗口管理**：滑动窗口、分层""")

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

md("""## 1. 上下文工程 (Context Engineering)

### 1.1 为什么 Context 是核心？

```
LLM 的能力取决于 context window:
  - 4K: 简单对话
  - 32K: 长文档
  - 128K: 代码库
  - 1M: 整本书

但 context window 有限:
  - 太多 → 超出窗口 (截断/报错)
  - 太少 → 模型"看不到"关键信息
  - 太杂 → "中间迷失" (lost in the middle)

上下文工程:
  在有限窗口内，放入最有价值的信息
  = 选择 + 压缩 + 组织
```

### 1.2 Context 的组成

```
Agent Context:
  ┌─────────────────────────┐
  │ System Prompt           │ ← 固定，定义角色
  │ 工具定义                 │ ← 固定，定义能力
  │ ─────────────────────── │
  │ 对话历史                 │ ← 增长，需要管理
  │ 工具调用结果              │ ← 增长，可能很大
  │ 检索内容                 │ ← 动态，按需加载
  │ ─────────────────────── │
  │ 当前用户输入             │ ← 最新
  └─────────────────────────┘
```

> 💡 上下文工程是 Agent 的"内存管理"——在有限窗口内放最有价值的信息。""")

md("""## 2. Context Loop

### 2.1 Agent 的上下文循环

```
Context Loop (Context Harness):
  while not done:
      1. 构建 context
         - system prompt
         - relevant history
         - tool results
         - retrieved info
      2. LLM(context) → action
      3. 执行 action → observation
      4. 更新 context
         - 添加 action + observation
         - 压缩/裁剪 if 超限
      5. 检查是否完成

关键: 每轮都要管理 context
      不是"一次性构建"
      而是"持续维护"
```""")

code("""# Context Loop 实现
class ContextLoop:
    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens
        self.system_prompt = ""
        self.history = []
        self.tool_results = []

    def estimate_tokens(self, text):
        # 粗略估计: 1 token ≈ 4 字符
        return len(text) // 4

    def build_context(self, user_input):
        parts = []

        # 1. System prompt (固定)
        parts.append(self.system_prompt)

        # 2. 对话历史 (可能压缩)
        history_text = "\\n".join(self.history)
        parts.append(history_text)

        # 3. 工具结果 (可能截断)
        tool_text = "\\n".join(self.tool_results)
        parts.append(tool_text)

        # 4. 当前输入
        parts.append(f"User: {user_input}")

        context = "\\n".join(parts)

        # 检查是否超限
        total_tokens = self.estimate_tokens(context)
        if total_tokens > self.max_tokens:
            context = self._compress(context)

        return context

    def _compress(self, context):
        # 简化压缩: 保留 system + 最近的历史
        lines = context.split("\\n")
        # 保留前 5 行 (system) 和后 10 行 (最近)
        if len(lines) > 15:
            compressed = lines[:5] + ["[...已压缩...]", *lines[-10:]]
            return "\\n".join(compressed)
        return context

    def update(self, action, observation):
        self.history.append(f"Action: {action}")
        self.tool_results.append(f"Result: {observation}")

# 模拟 Context Loop
loop = ContextLoop(max_tokens=200)
loop.system_prompt = "你是一个有用的助手。"

print("Context Loop 演示:")
print("=" * 50)

for i in range(5):
    user_input = f"第{i+1}个问题"
    context = loop.build_context(user_input)
    tokens = loop.estimate_tokens(context)

    action = f"回答{user_input}"
    observation = f"结果{i+1}"

    loop.update(action, observation)
    print(f"\\n轮 {i+1}: 输入='{user_input}', context={tokens} tokens")
    if tokens > 150:
        print("  ⚠️ context 接近上限，已压缩")""")

md("""## 3. Context 压缩

### 3.1 压缩策略

```
1. 摘要压缩 (Summarization)
   - 旧对话 → LLM 生成摘要
   - 摘要 + 最近对话 = 新 context

2. 选择性保留
   - 保留: 关键决策、工具结果
   - 丢弃: 寒暄、重复、错误

3. 结构化压缩
   - 把对话转为结构化笔记
   - "用户问了X，我回答了Y，结果是Z"

4. 分层压缩
   - 远期: 高度压缩 (摘要)
   - 中期: 轻度压缩 (关键点)
   - 近期: 不压缩 (完整)
```

### 3.2 摘要压缩示例

```
原始 (1000 tokens):
  User: 帮我分析公司财务
  Assistant: 好的，让我搜索...
  Tool: 2024年营收1亿...
  User: 利润呢？
  Assistant: 搜索利润...
  Tool: 利润2000万...
  User: 和去年比？
  ...

压缩后 (200 tokens):
  [摘要: 分析了公司财务，2024营收1亿，
   利润2000万，正在与去年对比]
  User: 和去年比？
  ...
```

> 💡 摘要压缩是 Agent 长对话的关键——让有限窗口容纳更多有效信息。""")

code("""# Context 压缩实现
class ContextCompressor:
    def __init__(self, max_tokens=1000):
        self.max_tokens = max_tokens

    def estimate_tokens(self, text):
        return len(text) // 4

    def summarize(self, text):
        # 模拟 LLM 摘要
        lines = text.split("\\n")
        key_points = [l for l in lines if any(kw in l for kw in ['结果', '决定', '关键'])]
        return "[摘要: " + "; ".join(key_points[:3]) + "]"

    def compress_history(self, history, keep_recent=3):
        if len(history) <= keep_recent:
            return history

        # 分层: 远期压缩 + 近期保留
        old = history[:-keep_recent]
        recent = history[-keep_recent:]

        old_text = "\\n".join(old)
        summary = self.summarize(old_text)

        return [summary] + recent

    def selective_keep(self, history):
        # 选择性保留: 只保留关键信息
        kept = []
        for msg in history:
            if any(kw in msg for kw in ['结果', '决定', '错误', '关键']):
                kept.append(msg)
        return kept

# 演示
compressor = ContextCompressor()

history = [
    "User: 你好",
    "Assistant: 你好！有什么可以帮你的？",
    "User: 搜索AI最新进展",
    "Assistant: 好的",
    "Tool结果: GPT-5发布了",
    "User: 详细说说",
    "Assistant: GPT-5的主要特点是...",
    "Tool结果: GPT-5支持100万token",
    "User: 和GPT-4比？",
    "Assistant: 让我比较一下",
    "Tool结果: GPT-5比GPT-4强50%",
]

print("原始历史 ({} 条):".format(len(history)))
for msg in history:
    print(f"  {msg}")

compressed = compressor.compress_history(history, keep_recent=3)
print(f"\\n压缩后 ({len(compressed)} 条):")
for msg in compressed:
    print(f"  {msg}")

kept = compressor.selective_keep(history)
print(f"\\n选择性保留 ({len(kept)} 条):")
for msg in kept:
    print(f"  {msg}")""")

md("""## 4. Context 路由

### 4.1 选择相关上下文

```
问题: Agent 有大量历史，但当前问题只需要一部分

Context 路由:
  1. 把历史分块 (chunks)
  2. 用 embedding 计算与当前问题的相似度
  3. 选 top-k 最相关的块
  4. 只放入相关的上下文

vs 全量放入:
  全量: 可能超出窗口 + "中间迷失"
  路由: 精准 + 高效
```

### 4.2 实现""")

code("""# Context 路由实现
class ContextRouter:
    def __init__(self):
        self.chunks = []
        self.embeddings = []

    def add_chunk(self, text, embedding):
        self.chunks.append(text)
        self.embeddings.append(embedding)

    def retrieve(self, query_embedding, top_k=3):
        # 计算相似度
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8)
            similarities.append(sim)

        # 选 top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.chunks[i], similarities[i]) for i in top_indices]

# 模拟历史分块和 embedding
np.random.seed(42)
router = ContextRouter()

chunks = [
    "讨论了公司财务: 营收1亿，利润2000万",
    "用户问了天气: 今天25度",
    "分析了代码bug: 空指针异常",
    "搜索了AI新闻: GPT-5发布",
    "计算了统计: 平均值50",
    "翻译了文档: 中英对照",
]

for chunk in chunks:
    emb = np.random.randn(64)  # 模拟 embedding
    router.add_chunk(chunk, emb)

# 当前问题: "公司利润是多少？"
query_emb = np.random.randn(64)  # 模拟 query embedding

results = router.retrieve(query_emb, top_k=3)

print("Context 路由结果:")
print("=" * 50)
print(f"当前问题: '公司利润是多少？'\\n")
for chunk, score in results:
    print(f"  相似度 {score:.4f}: {chunk}")
print("\\n→ 只把最相关的上下文放入 context window")""")

md("""## 5. 中间迷失问题

### 5.1 Lost in the Middle

```
现象:
  LLM 对 context 开头和结尾的信息关注多
  对中间的信息关注少

  [高关注] info1, info2, ..., [低关注] infoN/2, ..., [高关注] infoN

影响:
  - 关键信息放中间 → 可能被忽略
  - 检索结果太多 → 中间的被忽略

解决:
  1. 重要信息放开头或结尾
  2. 减少不必要的信息
  3. 重排 (reorder) 把相关的放两端
```""")

code("""# 中间迷失可视化
fig, ax = plt.subplots(figsize=(12, 5))

positions = np.arange(1, 21)
# 注意力分布: U 形 (开头和结尾高，中间低)
attention = 0.8 * np.exp(-positions / 3) + 0.8 * np.exp(-(20 - positions) / 3) + 0.2

ax.bar(positions, attention, color='steelblue', alpha=0.8, edgecolor='black')
ax.set_xlabel('Context 中的位置', fontsize=12)
ax.set_ylabel('模型关注度', fontsize=12)
ax.set_title('Lost in the Middle: 位置 vs 关注度', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 标注
ax.annotate('高关注', xy=(2, 0.9), fontsize=12, color='green', fontweight='bold')
ax.annotate('低关注', xy=(10, 0.25), fontsize=12, color='red', fontweight='bold')
ax.annotate('高关注', xy=(17, 0.9), fontsize=12, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('notebooks/fig_lost_in_middle.png', bbox_inches='tight')
plt.show()
print("重要信息放两端 → 避免中间迷失。")""")

md("""## 6. Context 窗口管理策略

### 6.1 策略对比

```
策略              描述              适用场景
滑动窗口           保留最近 N 轮      简单对话
摘要+近期          远期摘要+近期完整   长对话
分层记忆           短期/中期/长期     复杂 Agent
路由检索           按相关性选择       知识密集型
混合              以上组合           生产系统
```

### 6.2 生产建议

```
1. System prompt: 固定，精简
2. 工具定义: 按需加载，不全放
3. 对话历史: 摘要 + 近期
4. 检索结果: 路由 top-k + 重排
5. 监控: 实时跟踪 token 使用
```""")

code("""# Context 管理策略对比
fig, ax = plt.subplots(figsize=(10, 6))

strategies = ['滑动窗口', '摘要+近期', '分层记忆', '路由检索', '混合']
effectiveness = [3, 6, 7, 8, 9]
complexity = [2, 4, 6, 5, 8]

x = np.arange(len(strategies))
width = 0.35

ax.bar(x - width/2, effectiveness, width, label='效果', color='steelblue', alpha=0.8)
ax.bar(x + width/2, complexity, width, label='实现复杂度', color='coral', alpha=0.8)

ax.set_xlabel('策略', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('Context 管理策略对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_context_strategies.png', bbox_inches='tight')
plt.show()
print("混合策略效果最好但最复杂; 摘要+近期性价比最高。")""")

md("""## 7. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 上下文工程 | ✅ |
| Context Loop | ✅ |
| Context 压缩 (摘要/选择性) | ✅ |
| Context 路由 (相关性检索) | ✅ |
| 中间迷失问题 | ✅ |
| 窗口管理策略 | ✅ |

### 核心 takeaway
> **上下文工程是 Agent 的内存管理**——在有限窗口内放最有价值的信息。压缩让历史不超限，路由让信息更精准，重排避免中间迷失。

### 🔗 下一章
**`55_rag_agentic.ipynb`** — RAG、Agentic RAG、GraphRAG

---

> 💬 **板块九(Agent 与系统)进行中 (3/9)。**""")

output_path = "notebooks/54_context_engineering.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")