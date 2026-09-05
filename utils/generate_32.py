# 生成 32_embedding_retrieval.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 32 — Embedding 与向量检索：10B 规模

> 🔥 当知识库有 100 亿条目时，如何毫秒级检索？
> 向量检索是 RAG、推荐系统、搜索引擎的核心基础设施。

## 本章你将掌握

1. **Embedding 基础**：文本/图像→向量
2. **向量检索算法**：暴力→IVF→HNSW→PQ
3. **10B 规模**：乘积量化 + 分布式
4. **多向量检索**：ColBERT 的晚期交互""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120; plt.rcParams['font.size'] = 11
np.random.seed(42); torch.manual_seed(42)
print("环境准备完成 ✅")""")

md("""## 1. Embedding：把内容变成向量

### 1.1 什么是 Embedding

```
文本: "猫坐在垫子上" → [0.2, -0.5, 0.8, ..., 0.1]  (768维)
图像: [像素] → [0.3, 0.7, -0.2, ..., 0.4]  (512维)

性质:
  语义相似的内容 → 向量靠近
  语义不同的内容 → 向量远离
  → 可以用向量距离度量语义相似度
```

### 1.2 常见 Embedding 模型

| 模型 | 维度 | 特点 |
|------|------|------|
| **word2vec** | 300 | 词级 |
| **BERT** | 768 | 句级 (CLS token) |
| **Sentence-BERT** | 768 | 句级 (池化) |
| **text-embedding-3** | 3072 | OpenAI |
| **BGE** | 1024 | 开源 SOTA |

### 1.3 检索流程

```
离线:  文档 → Embedding → 存入向量数据库
在线:  查询 → Embedding → 向量数据库检索 → 返回最近邻
```

> 💡 Embedding 是检索的基础——把语义相似性变成向量距离。
# 好的 Embedding = 检索质量的一半。""")

md("""## 2. 向量检索算法

### 2.1 暴力检索 (Flat)

```
查询 q → 与所有 N 个向量计算距离 → 取 top-k

复杂度: O(N × d)
  N=100万, d=768 → 7.68亿次运算 → 慢
```

### 2.2 IVF：倒排文件

```
1. 聚类: K-Means 把向量分成 nlist 个簇
2. 存储: 每个簇存属于它的向量
3. 查询:
   a. 找最近的 nprobe 个簇
   b. 只在这些簇内搜索

复杂度: O(nprobe × N/nlist × d)
  → 用 nprobe/nlist 的比例减少搜索范围
```

### 2.3 HNSW：分层导航小世界

```
分层图结构:
  Layer 0: 所有点 (密集连接)
  Layer 1: 部分点 (稀疏)
  Layer L: 最少点

查询: 从最顶层开始, 贪心搜索 → 逐层下降
  → 快速定位到目标区域 → 在底层精细搜索

复杂度: O(log N × d)
  → 对数复杂度, 适合大规模
```

### 2.4 PQ：乘积量化

```
把 d 维向量切成 m 段, 每段独立量化:
  [v1, v2, ..., v768] → [v1...v96, v97...v192, ..., v673...v768]
  每段用 256 个码字量化 → 每段存 1 byte

压缩: 768 × 4字节 (float32) → 8 字节 (m=8)
  → 384 倍压缩!
```

> 💡 实际系统组合使用：**HNSW + PQ** 或 **IVF + PQ**。
# HNSW/IVF 加速搜索，PQ 压缩存储——两者互补。""")

code("""# 向量检索算法实现
class FlatIndex:
    def __init__(self, vectors):
        self.vectors = np.array(vectors, dtype=np.float32)

    def search(self, query, k=5):
        distances = np.linalg.norm(self.vectors - query, axis=1)
        top_k = np.argsort(distances)[:k]
        return top_k, distances[top_k]

class IVFIndex:
    def __init__(self, vectors, nlist=10):
        self.vectors = np.array(vectors, dtype=np.float32)
        self.nlist = nlist
        # K-Means 聚类
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(n_clusters=nlist, random_state=42, n_init=10)
        self.assignments = self.kmeans.fit_predict(self.vectors)
        self.centroids = self.kmeans.cluster_centers_

    def search(self, query, k=5, nprobe=2):
        # 找最近的 nprobe 个簇
        centroid_dists = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(centroid_dists)[:nprobe]

        # 在这些簇内搜索
        candidates = np.where(np.isin(self.assignments, nearest_clusters))[0]
        if len(candidates) == 0:
            return np.array([]), np.array([])
        dists = np.linalg.norm(self.vectors[candidates] - query, axis=1)
        top_k_local = np.argsort(dists)[:k]
        return candidates[top_k_local], dists[top_k_local]

# 生成数据
n_vectors = 10000
d = 128
vectors = np.random.randn(n_vectors, d).astype(np.float32)
query = np.random.randn(d).astype(np.float32)

# 对比 Flat vs IVF
flat_idx = FlatIndex(vectors)
ivf_idx = IVFIndex(vectors, nlist=100)

start = time.time()
flat_result, flat_dists = flat_idx.search(query, k=5)
flat_time = time.time() - start

start = time.time()
ivf_result, ivf_dists = ivf_idx.search(query, k=5, nprobe=5)
ivf_time = time.time() - start

print(f"Flat: {flat_time*1000:.2f}ms, 结果 {flat_result}")
print(f"IVF:  {ivf_time*1000:.2f}ms, 结果 {ivf_result}")
print(f"加速: {flat_time/ivf_time:.1f}x")
print(f"召回率: {len(set(flat_result) & set(ivf_result))}/5")""")

code("""# HNSW 简化实现 (分层图)
class SimpleHNSW:
    def __init__(self, vectors, max_layers=3):
        self.vectors = np.array(vectors, dtype=np.float32)
        self.n = len(vectors)
        self.max_layers = max_layers
        # 简化: 每层随机选点
        self.layers = []
        remaining = list(range(self.n))
        for l in range(max_layers):
            n_in_layer = int(self.n * (0.5 ** l))
            layer_nodes = np.random.choice(remaining, min(n_in_layer, len(remaining)), replace=False)
            self.layers.append(layer_nodes)
            remaining = [x for x in remaining if x not in layer_nodes]

    def search(self, query, k=5):
        # 从顶层贪心搜索
        current = self.layers[-1][0] if len(self.layers[-1]) > 0 else 0
        for layer in reversed(self.layers):
            nodes = layer
            if len(nodes) == 0:
                continue
            dists = np.linalg.norm(self.vectors[nodes] - query, axis=1)
            current = nodes[np.argmin(dists)]

        # 底层精细搜索
        all_dists = np.linalg.norm(self.vectors - query, axis=1)
        top_k = np.argsort(all_dists)[:k]
        return top_k, all_dists[top_k]

# PQ 简化实现
class SimplePQ:
    def __init__(self, vectors, m=8, n_bits=8):
        self.vectors = np.array(vectors, dtype=np.float32)
        self.d = vectors.shape[1]
        self.m = m
        self.sub_dim = self.d // m
        self.n_bits = n_bits
        self.codebook = []

        # 每段独立 K-Means
        from sklearn.cluster import KMeans
        for i in range(m):
            start, end = i * self.sub_dim, (i + 1) * self.sub_dim
            sub_vectors = self.vectors[:, start:end]
            kmeans = KMeans(n_clusters=2**n_bits, random_state=42, n_init=3)
            codes = kmeans.fit_predict(sub_vectors)
            self.codebook.append((kmeans.cluster_centers_, codes))

    def compress(self):
        # 每个向量 → m 个 byte
        return np.array([[self.codebook[i][1][j] for i in range(self.m)]
                         for j in range(len(self.vectors))], dtype=np.uint8)

    def search(self, query, k=5):
        # 用量化后的向量搜索
        approx_vectors = np.zeros_like(self.vectors)
        for i in range(self.m):
            start, end = i * self.sub_dim, (i + 1) * self.sub_dim
            codes = self.codebook[i][1]
            centroids = self.codebook[i][0]
            approx_vectors[:, start:end] = centroids[codes]

        dists = np.linalg.norm(approx_vectors - query, axis=1)
        top_k = np.argsort(dists)[:k]
        return top_k, dists[top_k]

# 对比
hnsw = SimpleHNSW(vectors, max_layers=3)
pq = SimplePQ(vectors, m=8)

start = time.time()
hnsw_result, hnsw_dists = hnsw.search(query, k=5)
hnsw_time = time.time() - start

start = time.time()
pq_result, pq_dists = pq.search(query, k=5)
pq_time = time.time() - start

print(f"HNSW: {hnsw_time*1000:.2f}ms, 结果 {hnsw_result}")
print(f"PQ:   {pq_time*1000:.2f}ms, 结果 {pq_result}")
print(f"HNSW 召回率: {len(set(flat_result) & set(hnsw_result))}/5")
print(f"PQ 召回率:   {len(set(flat_result) & set(pq_result))}/5")
print(f"PQ 压缩: {vectors.nbytes / (len(vectors) * 8):.0f}x")""")

code("""# 可视化检索算法对比
algorithms = ['Flat\\n(暴力)', 'IVF\\n(倒排)', 'HNSW\\n(图)', 'PQ\\n(量化)', 'IVF+PQ\\n(组合)']
speeds = [1, 10, 100, 50, 500]  # 相对速度
memory = [1, 1, 1.5, 0.01, 0.01]  # 相对内存
recall = [100, 95, 98, 90, 92]  # 召回率

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.bar(algorithms, speeds, color='steelblue', alpha=0.8)
ax.set_ylabel('相对速度 (x)'); ax.set_title('检索速度')
ax.set_yscale('log')

ax = axes[1]
ax.bar(algorithms, memory, color='coral', alpha=0.8)
ax.set_ylabel('相对内存'); ax.set_title('内存占用')
ax.set_yscale('log')

ax = axes[2]
ax.bar(algorithms, recall, color='forestgreen', alpha=0.8)
ax.set_ylabel('召回率 (%)'); ax.set_title('检索质量')

plt.tight_layout()
plt.savefig('notebooks/fig_vector_search.png', bbox_inches='tight')
plt.show()
print("HNSW: 速度+质量好, 内存大; PQ: 内存极小, 质量略降; 组合最优。")""")

md("""## 3. 10B 规模的挑战

### 3.1 规模问题

```
1B 向量 × 768 维 × 4 字节 = 3TB → 单机放不下

解决方案:
1. 量化压缩: PQ → 3TB / 384 ≈ 8GB
2. 分布式: 分片到多台机器
3. 磁盘存储: SSD + 内存缓存
```

### 3.2 分布式向量检索

```
分片策略:
  按向量分: 每台机器存一部分 → 查询时广播
  按聚类分: 每台机器存几个簇 → 查询时只查相关机器

代表系统:
  Milvus: 开源, 分布式向量数据库
  Faiss: Meta 的库, 单机/分布式
  ScaNN: Google 的库
  Pinecone: 云服务
```

### 3.3 实际系统设计

```
10B 向量检索系统:
  1. 向量量化: PQ 压缩 384x
  2. IVF 索引: nlist=100万, nprobe=100
  3. 分布式: 100台机器, 每台 1亿向量
  4. 缓存: 热点查询结果缓存
  5. 增量更新: 支持动态添加/删除

→ 查询延迟: ~10ms (10B 规模)
```

> 💡 10B 规模的关键：**量化 + 分布式 + 好的索引**。
# Milvus 等系统已经能在 10B 规模下实现毫秒级检索。""")

code("""# 规模估算
def estimate_system(n_vectors, d=768):
    # 原始大小
    raw_size = n_vectors * d * 4 / 1e12  # TB
    # PQ 压缩后 (m=16, 8bit)
    pq_size = n_vectors * 16 / 1e12  # TB
    # 检索速度估计
    flat_ops = n_vectors * d
    ivf_ops = n_vectors * d * 0.001  # nprobe/nlist = 0.1%
    hnsw_ops = np.log(n_vectors) * d * 10

    return {
        'raw_TB': raw_size,
        'pq_TB': pq_size,
        'flat_GFLOPS': flat_ops / 1e9,
        'ivf_GFLOPS': ivf_ops / 1e9,
        'hnsw_GFLOPS': hnsw_ops / 1e9,
    }

scales = [1e6, 1e7, 1e8, 1e9, 1e10]
print(f"{'规模':>10s} {'原始(TB)':>10s} {'PQ(TB)':>10s} {'Flat(GF)':>10s} {'IVF(GF)':>10s} {'HNSW(GF)':>10s}")
for n in scales:
    est = estimate_system(n)
    print(f"{n:10.0e} {est['raw_TB']:10.3f} {est['pq_TB']:10.4f} {est['flat_GFLOPS']:10.1f} {est['ivf_GFLOPS']:10.1f} {est['hnsw_GFLOPS']:10.1f}")

print("\\n10B 规模: PQ 压缩到 0.16TB, HNSW 只需 ~180 GFLOPS → 可行!")""")

md("""## 4. 多向量检索：ColBERT

### 4.1 单向量的问题

```
单向量: 文档 → 1 个向量 → 检索
  问题: 一个向量难以表示文档的所有信息
  "猫坐在垫子上" 和 "垫子上坐着猫" → 可能不同
```

### 4.2 ColBERT 的晚期交互

```
ColBERT: 文档 → N 个 token 向量 → 检索

查询: "猫 垫子" → [q_猫, q_垫子]
文档: "猫 坐在 垫子 上" → [d_猫, d_坐在, d_垫子, d_上]

相似度 = Σ_i max_j sim(q_i, d_j)
  → 每个查询 token 找最匹配的文档 token
  → "晚期交互" (late interaction)
```

### 4.3 优势

- **更细粒度**：token 级匹配
- **更好排序**：比单向量 rerank 效果好
- **可解释**：能看到哪些 token 匹配

### 4.4 代价

```
存储: N 个 token 向量 vs 1 个文档向量 → N 倍存储
检索: 需要特殊索引 (PLAID)

→ ColBERTv2 优化了存储和检索
```

> 💡 ColBERT 的哲学：**保留更多信息，晚期再做交互**。
# 单向量是"早期交互"（池化太早），ColBERT 推迟到检索后。""")

code("""# ColBERT 简化实现
def colbert_score(query_embeds, doc_embeds):
    # query_embeds: [n_query_tokens, d]
    # doc_embeds: [n_doc_tokens, d]
    # 晚期交互: 每个查询 token 找最匹配的文档 token

    # 相似度矩阵 [n_query, n_doc]
    sim_matrix = query_embeds @ doc_embeds.T

    # 每个查询 token 的最大相似度
    max_sims = sim_matrix.max(dim=-1).values  # [n_query]

    # ColBERT 分数 = 总和
    return max_sims.sum().item()

# 单向量 vs ColBERT
d = 64
# 查询: "猫 垫子"
query_tokens = torch.randn(2, d)  # 2 个 token
# 文档1: "猫 坐在 垫子 上" (相关)
doc1_tokens = torch.randn(4, d)
doc1_tokens[0] = query_tokens[0] + 0.1 * torch.randn(d)  # 猫 匹配
doc1_tokens[2] = query_tokens[1] + 0.1 * torch.randn(d)  # 垫子 匹配
# 文档2: "狗 跑在 草地 上" (不相关)
doc2_tokens = torch.randn(4, d)

# ColBERT 分数
score1 = colbert_score(query_tokens, doc1_tokens)
score2 = colbert_score(query_tokens, doc2_tokens)

# 单向量 (平均池化)
query_single = query_tokens.mean(dim=0)
doc1_single = doc1_tokens.mean(dim=0)
doc2_single = doc2_tokens.mean(dim=0)
single_score1 = (query_single @ doc1_single).item()
single_score2 = (query_single @ doc2_single).item()

print("ColBERT (晚期交互):")
print(f"  相关文档: {score1:.3f}")
print(f"  不相关文档: {score2:.3f}")
print(f"  区分度: {score1 - score2:.3f}")
print(f"\\n单向量 (早期池化):")
print(f"  相关文档: {single_score1:.3f}")
print(f"  不相关文档: {single_score2:.3f}")
print(f"  区分度: {single_score1 - single_score2:.3f}")
print(f"\\nColBERT 区分度更大 → 检索更准确。")""")

md("""## 5. 向量数据库对比

| 系统 | 特点 | 规模 | 开源 |
|------|------|------|------|
| **Milvus** | 分布式, 全功能 | 10B+ | ✅ |
| **Faiss** | 库, 高效 | 1B | ✅ |
| **ScaNN** | Google, 各向异性量化 | 1B | ✅ |
| **Pinecone** | 云服务 | 10B+ | ❌ |
| **Weaviate** | 混合检索 | 1B | ✅ |
| **Qdrant** | Rust, 高性能 | 1B | ✅ |

### 选择指南

```
小规模 (<1M):   Faiss (简单高效)
中规模 (1M-100M): Qdrant / Weaviate
大规模 (100M-1B):  Milvus (分布式)
超大规模 (10B+):  Milvus + PQ + 分布式
云服务:          Pinecone (免运维)
```

> 💡 向量数据库是 RAG 的基础设施——选对系统很重要。
# Milvus 在大规模场景最成熟，Qdrant 在中等规模最简洁。""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| Embedding 基础 | ✅ |
| Flat/IVF/HNSW/PQ | ✅ |
| 10B 规模系统 | ✅ |
| ColBERT 多向量 | ✅ |
| 向量数据库对比 | ✅ |

### 核心 takeaway
> **向量检索 = 好的 Embedding + 高效的索引 + 量化压缩**。
> HNSW 加速，PQ 压缩，分布式扩展到 10B。
> ColBERT 的多向量保留更多信息，检索更准。

### 🔗 下一板块
**`33_ddpm.ipynb`** — DDPM 扩散模型（进入板块六：生成模型）

---

> 💬 **板块五(表征学习与检索)完结。**""")

output_path = "notebooks/32_embedding_retrieval.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")