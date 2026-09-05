# 生成 55_rag_agentic.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 55 — RAG 与 Agentic RAG、GraphRAG

> 🔥 RAG 让 LLM 能"查资料"，Agentic RAG 让它"研究会"，GraphRAG 让它"理解关系"。

## 本章你将掌握

1. **Naive RAG**：检索-增强-生成
2. **Advanced RAG**：查询改写、重排
3. **Agentic RAG**：Agent 驱动的迭代检索
4. **GraphRAG**：知识图谱 + RAG""")

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

md("""## 1. Naive RAG

### 1.1 基本流程

```
RAG (Retrieval-Augmented Generation):
  1. 用户提问
  2. 把问题转为 embedding
  3. 在向量库中检索 top-k 相关文档
  4. 把检索结果 + 问题拼接 → LLM
  5. LLM 基于检索结果生成回答

  Question → Embed → Search → Retrieve → Augment → Generate
```

### 1.2 为什么需要 RAG？

```
纯 LLM 的问题:
  - 知识截止训练时 (不知道最新)
  - 可能幻觉 (编造信息)
  - 不知道私有数据

RAG 的优势:
  - 实时: 检索最新信息
  - 可溯源: 答案有出处
  - 私有: 检索企业内部数据
  - 省钱: 不用微调就能加知识
```

> 💡 RAG = 给 LLM 一个"图书馆"——先查资料再回答。""")

code("""# Naive RAG 实现
class NaiveRAG:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs, embs):
        self.documents.extend(docs)
        self.embeddings.extend(embs)

    def retrieve(self, query_emb, top_k=3):
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
            similarities.append(sim)
        top_idx = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_idx]

    def generate(self, query, retrieved):
        # 模拟 LLM 生成
        context = "\\n".join([doc for doc, _ in retrieved])
        return f"基于以下信息回答 '{query}':\\n{context}\\n→ 答案: 根据资料..."

# 模拟文档库
np.random.seed(42)
rag = NaiveRAG()

docs = [
    "GPT-5 于2025年发布，支持100万token上下文。",
    "Claude 3.5 在代码生成上表现优异。",
    "Llama 3 是开源大模型，性能接近闭源。",
    "Diffusion 模型用于图像生成。",
    "Transformer 架构是现代NLP的基础。",
]

for doc in docs:
    emb = np.random.randn(128)
    rag.add_documents([doc], [emb])

# 查询
query = "GPT-5的上下文长度是多少？"
query_emb = np.random.randn(128)

retrieved = rag.retrieve(query_emb, top_k=3)
answer = rag.generate(query, retrieved)

print("Naive RAG:")
print("=" * 50)
print(f"查询: {query}")
print(f"\\n检索结果:")
for doc, score in retrieved:
    print(f"  [{score:.4f}] {doc}")
print(f"\\n生成: {answer}")""")

md("""## 2. Advanced RAG

### 2.1 改进点

```
Naive RAG 的问题:
  - 查询可能不匹配文档表述
  - 检索结果可能有噪声
  - top-k 可能不够或太多

Advanced RAG 改进:
  1. 查询改写 (Query Rewriting)
  2. 多查询融合 (Multi-Query)
  3. 重排 (Reranking)
  4. 混合检索 (Dense + Sparse)
```

### 2.2 查询改写

```
原始查询: "它的性能怎么样"
  → "它"指代不明 → 改写

改写后: "GPT-5的性能怎么样"
  → 明确指代 → 检索更准

多查询:
  原始: "GPT-5性能"
  改写1: "GPT-5 benchmark结果"
  改写2: "GPT-5 vs GPT-4对比"
  → 三个查询分别检索 → 合并结果
```

### 2.3 重排

```
检索: 向量相似度 (快但不精确)
  → top-20 候选

重排: Cross-encoder (慢但精确)
  → top-5 最终

两阶段: 快速召回 + 精确重排
```

> 💡 Advanced RAG = 查询改写 + 多路召回 + 精确重排——每一步都在提升精度。""")

code("""# Advanced RAG 实现
class AdvancedRAG:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs, embs):
        self.documents.extend(docs)
        self.embeddings.extend(embs)

    def rewrite_query(self, query):
        # 模拟查询改写
        rewrites = [query]
        if "性能" in query:
            rewrites.append(query.replace("性能", "benchmark"))
            rewrites.append(query.replace("性能", "评测结果"))
        return rewrites

    def multi_retrieve(self, query_embs, top_k=5):
        all_results = []
        for emb in query_embs:
            for i, doc_emb in enumerate(self.embeddings):
                sim = np.dot(emb, doc_emb) / (
                    np.linalg.norm(emb) * np.linalg.norm(doc_emb) + 1e-8)
                all_results.append((self.documents[i], sim))
        # 去重 + 排序
        seen = set()
        unique = []
        for doc, sim in sorted(all_results, key=lambda x: -x[1]):
            if doc not in seen:
                seen.add(doc)
                unique.append((doc, sim))
        return unique[:top_k]

    def rerank(self, query, candidates, top_k=3):
        # 模拟 cross-encoder 重排
        reranked = []
        for doc, sim in candidates:
            # 重排分数 = 检索分数 + 关键词匹配
            keyword_score = sum(1 for w in query if w in doc) / max(len(query), 1)
            final_score = 0.7 * sim + 0.3 * keyword_score
            reranked.append((doc, final_score))
        reranked.sort(key=lambda x: -x[1])
        return reranked[:top_k]

# 演示
np.random.seed(42)
adv_rag = AdvancedRAG()

for doc in docs:
    emb = np.random.randn(128)
    adv_rag.add_documents([doc], [emb])

query = "GPT-5性能怎么样？"

# 1. 查询改写
rewrites = adv_rag.rewrite_query(query)
print("1. 查询改写:")
for i, rw in enumerate(rewrites):
    print(f"  {i+1}. {rw}")

# 2. 多路检索
query_embs = [np.random.randn(128) for _ in rewrites]
candidates = adv_rag.multi_retrieve(query_embs, top_k=5)
print(f"\\n2. 多路检索 ({len(candidates)} 候选):")
for doc, score in candidates:
    print(f"  [{score:.4f}] {doc[:30]}...")

# 3. 重排
final = adv_rag.rerank(query, candidates, top_k=3)
print(f"\\n3. 重排后 (top-3):")
for doc, score in final:
    print(f"  [{score:.4f}] {doc[:30]}...")""")

md("""## 3. Agentic RAG

### 3.1 从被动到主动

```
Naive RAG: 被动
  检索一次 → 生成 → 结束

Agentic RAG: 主动
  Agent 决定:
  - 是否需要检索?
  - 检索什么?
  - 结果够不够?
  - 需要再检索吗?
  - 何时停止?

循环:
  while not enough:
      1. 分析当前信息
      2. 决定下一步检索
      3. 检索 + 更新
      4. 评估是否足够
```

### 3.2 vs Naive RAG

```
          Naive RAG        Agentic RAG
检索次数   1               多次 (按需)
查询       原始查询         动态改写
决策       固定流程         Agent 自主
适用       简单事实          复杂研究
```

> 💡 Agentic RAG 把检索从"一次性"变成"迭代式"——Agent 自主决定何时、检索什么。""")

code("""# Agentic RAG 实现
class AgenticRAG:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.search_history = []

    def retrieve(self, query_emb, top_k=3):
        sims = [np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
                for emb in self.embeddings]
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [(self.documents[i], sims[i]) for i in top_idx]

    def think(self, query, retrieved):
        # 模拟 Agent 思考
        info_count = len(retrieved)
        if info_count < 2:
            return "need_more", "信息不够，需要更多检索"
        elif any("GPT" in doc for doc, _ in retrieved):
            return "enough", "信息足够，可以回答"
        else:
            return "need_more", "需要更相关的信息"

    def run(self, query, max_rounds=3):
        print(f"Agentic RAG: '{query}'")
        print("=" * 50)

        collected_info = []

        for round_num in range(max_rounds):
            # 1. 生成查询 (可能改写)
            search_query = query if round_num == 0 else f"更多关于{query}"
            query_emb = np.random.randn(128)

            # 2. 检索
            retrieved = self.retrieve(query_emb, top_k=3)
            self.search_history.append(search_query)
            collected_info.extend(retrieved)

            print(f"\\n轮 {round_num + 1}: 检索 '{search_query}'")
            for doc, score in retrieved:
                print(f"  [{score:.4f}] {doc[:40]}...")

            # 3. 思考: 信息够吗？
            decision, reason = self.think(query, collected_info)
            print(f"  → {reason}")

            if decision == "enough":
                print(f"\\n✅ 完成! 共检索 {len(self.search_history)} 次")
                return collected_info

        print(f"\\n达到最大轮数 {max_rounds}")
        return collected_info

# 演示
np.random.seed(42)
embs = [np.random.randn(128) for _ in docs]
agentic_rag = AgenticRAG(docs, embs)
result = agentic_rag.run("GPT-5的上下文长度", max_rounds=3)""")

md("""## 4. GraphRAG

### 4.1 从向量到图

```
Naive RAG: 向量检索
  - 检索独立文档片段
  - 不理解文档间关系
  - 全局性问题难回答

GraphRAG: 知识图谱 + RAG
  1. 从文档构建知识图谱
     实体 → 节点
     关系 → 边
  2. 社区检测 → 层次摘要
  3. 查询时:
     局部问题 → 检索相关子图
     全局问题 → 用社区摘要
```

### 4.2 GraphRAG 流程

```
构建阶段:
  文档 → LLM 抽取实体/关系 → 知识图谱
  图 → 社区检测 (Leiden) → 层次摘要
  每层社区 → LLM 生成摘要

查询阶段:
  局部查询: "X是什么?"
    → 找 X 的节点 → 检索邻居子图

  全局查询: "文档的主要主题?"
    → 遍历顶层社区摘要 → 合并
```

> 💡 GraphRAG 能回答全局性问题——这是 Naive RAG 做不到的。""")

code("""# GraphRAG 简化实现
class GraphRAG:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.communities = {}
        self.summaries = {}

    def add_entity(self, name, entity_type, description):
        self.nodes[name] = {'type': entity_type, 'desc': description}

    def add_relation(self, src, dst, relation):
        self.edges.append((src, dst, relation))

    def detect_communities(self):
        # 简化: 按类型分组
        communities = {}
        for node, info in self.nodes.items():
            t = info['type']
            if t not in communities:
                communities[t] = []
            communities[t].append(node)
        self.communities = communities
        return communities

    def summarize_community(self, community_name):
        nodes = self.communities[community_name]
        descs = [f"{n}: {self.nodes[n]['desc']}" for n in nodes]
        summary = f"社区 '{community_name}': " + "; ".join(descs)
        self.summaries[community_name] = summary
        return summary

    def local_query(self, entity):
        # 局部查询: 找实体 + 邻居
        if entity not in self.nodes:
            return "未找到实体"
        result = [f"{entity}: {self.nodes[entity]['desc']}"]
        for src, dst, rel in self.edges:
            if src == entity:
                result.append(f"  →{rel}→ {dst}: {self.nodes[dst]['desc']}")
            elif dst == entity:
                result.append(f"  ←{rel}← {src}: {self.nodes[src]['desc']}")
        return "\\n".join(result)

    def global_query(self):
        # 全局查询: 用社区摘要
        return "\\n".join(self.summaries.values())

# 构建 GraphRAG
graphrag = GraphRAG()

# 添加实体
graphrag.add_entity("GPT-5", "模型", "OpenAI的最新模型")
graphrag.add_entity("OpenAI", "公司", "AI研究公司")
graphrag.add_entity("Claude", "模型", "Anthropic的模型")
graphrag.add_entity("Anthropic", "公司", "AI安全公司")
graphrag.add_entity("Transformer", "技术", "基础架构")

# 添加关系
graphrag.add_relation("GPT-5", "OpenAI", "由...开发")
graphrag.add_relation("Claude", "Anthropic", "由...开发")
graphrag.add_relation("GPT-5", "Transformer", "基于")
graphrag.add_relation("Claude", "Transformer", "基于")

# 社区检测
communities = graphrag.detect_communities()
for name, nodes in communities.items():
    graphrag.summarize_community(name)

print("GraphRAG:")
print("=" * 50)
print(f"\\n实体: {list(graphrag.nodes.keys())}")
print(f"关系: {graphrag.edges}")
print(f"社区: {communities}")

print(f"\\n--- 局部查询 ---")
print(graphrag.local_query("GPT-5"))

print(f"\\n--- 全局查询 ---")
print(graphrag.global_query())""")

md("""## 5. RAG 范式对比

### 5.1 总结

```
范式            检索      适用场景
Naive RAG      一次      简单事实问答
Advanced RAG   改写+重排  精度要求高
Agentic RAG    迭代      复杂研究
GraphRAG       图+社区    全局性问题
Self-RAG       自判断     减少幻觉
Modular RAG    模块化     灵活定制
```""")

code("""# RAG 范式对比
fig, ax = plt.subplots(figsize=(12, 6))

paradigms = ['Naive\\nRAG', 'Advanced\\nRAG', 'Agentic\\nRAG', 'Graph\\nRAG', 'Self\\nRAG']
accuracy = [5, 7, 8, 9, 7]
cost = [2, 4, 6, 8, 5]
latency = [2, 4, 7, 8, 5]

x = np.arange(len(paradigms))
width = 0.25

ax.bar(x - width, accuracy, width, label='准确率', color='steelblue', alpha=0.8)
ax.bar(x, cost, width, label='成本', color='coral', alpha=0.8)
ax.bar(x + width, latency, width, label='延迟', color='forestgreen', alpha=0.8)

ax.set_xlabel('RAG 范式', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('RAG 范式对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(paradigms, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_rag_paradigms.png', bbox_inches='tight')
plt.show()
print("GraphRAG 准确率最高但最贵; Naive RAG 最快最便宜。")""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| Naive RAG | ✅ |
| Advanced RAG (改写+重排) | ✅ |
| Agentic RAG (迭代检索) | ✅ |
| GraphRAG (知识图谱+社区) | ✅ |
| 范式对比 | ✅ |

### 核心 takeaway
> **RAG 从被动到主动、从向量到图**——Naive RAG 一次检索，Agentic RAG 迭代研究，GraphRAG 理解全局结构。选择取决于任务复杂度。

### 🔗 下一章
**`56_memory_systems.ipynb`** — 记忆系统、MemGPT/Letta

---

> 💬 **板块九(Agent 与系统)进行中 (4/9)。**""")

output_path = "notebooks/55_rag_agentic.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")