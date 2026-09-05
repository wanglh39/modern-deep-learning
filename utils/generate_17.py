"""生成 17_retrieval_augmented_training.ipynb 的脚本"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))

# ============================================================
md("""# 17 — 检索增强训练：RETRO / kNN-LM

> 与其把所有知识塞进参数，不如**外挂一个数据库**。
> 检索增强训练让模型在生成时**查阅数据库**——用更少参数达到更好效果。
> 这和 RAG 不同：RAG 在推理时检索，检索增强训练在**训练时就引入检索**。

## 本章你将掌握

1. **检索增强语言模型**的概念
2. **kNN-LM**：k近邻语言模型
3. **RETRO**：检索增强 Transformer
4. **检索 vs 长上下文**的对比""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
torch.manual_seed(42)
print("环境准备完成 ✅")""")

# ============================================================
md("""## 1. 检索增强语言模型

### 1.1 动机：参数不是唯一的知识存储

```
普通 LM:  所有知识存在参数里 → 参数要多 → 训练贵
检索 LM:  常见知识存数据库, 参数只学推理 → 参数少 → 训练便宜
```

### 1.2 训练时检索 vs 推理时检索

| | RAG (推理时) | RETRO (训练时) |
|---|-------------|----------------|
| **检索时机** | 推理时查 | 训练+推理都查 |
| **模型学习** | 不学检索 | 学会利用检索结果 |
| **融合方式** | prompt 拼接 | cross-attention |

> 💡 RETRO 的关键：模型在**训练时就学会利用检索结果**，而不是推理时才拼接。
> 这让模型更深度地融合外部知识。""")

# ============================================================
md("""## 2. kNN-LM：k近邻语言模型

### 2.1 kNN-LM 的思想

kNN-LM 最简单——不改模型，在推理时**用 kNN 检索覆盖部分预测**：

```
1. 把训练数据的 (context, next_token) 存入向量数据库
2. 推理时:
   a. 用模型预测 P_model(token | context)
   b. 用 kNN 检索相似 context 的 next_token
   c. 混合: P = λ * P_model + (1-λ) * P_kNN
```

### 2.2 优势

- **零训练**：直接在已有 LM 上外挂
- **可更新**：更新数据库 = 更新知识，不用重训
- **长尾知识**：罕见事实存在数据库比参数好

> 💡 kNN-LM 证明：很多知识在 LM 的**隐状态**里，可以直接检索。""")

code("""# kNN-LM 简化实现
class SimpleKNNLM:
    # kNN-LM: 模型预测 + kNN检索
    def __init__(self, vocab_size, embed_dim=16, k=5, lambda_mix=0.7):
        self.vocab_size = vocab_size
        self.k = k
        self.lambda_mix = lambda_mix
        # 简化: 用随机嵌入当"模型"
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        # kNN 数据库: (key, value) = (context_embedding, next_token)
        self.db_keys = None
        self.db_values = None

    def build_database(self, sequences):
        # 从序列构建 (context_emb, next_token) 数据库
        keys, values = [], []
        for seq in sequences:
            for i in range(len(seq) - 1):
                ctx_emb = self.embed(torch.tensor(seq[i])).detach()
                keys.append(ctx_emb)
                values.append(seq[i + 1])
        self.db_keys = torch.stack(keys)
        self.db_values = torch.tensor(values)

    def model_predict(self, token_id):
        # 模型预测下一个 token 的分布
        emb = self.embed(torch.tensor([token_id]))
        logits = self.head(emb)
        return F.softmax(logits, dim=-1)

    def knn_retrieve(self, token_id, k=None):
        # kNN 检索: 找最相似的 k 个 context, 用它们的 next_token 投票
        if k is None:
            k = self.k
        query = self.embed(torch.tensor([token_id])).detach()
        # 计算相似度
        sims = F.cosine_similarity(query, self.db_keys)
        # 取 top-k
        topk_sims, topk_idx = sims.topk(k)
        topk_tokens = self.db_values[topk_idx]
        # 构建分布
        probs = torch.zeros(self.vocab_size)
        for t, s in zip(topk_tokens, topk_sims):
            probs[t] += torch.exp(s)
        return probs / (probs.sum() + 1e-8)

    def predict(self, token_id):
        # 混合: λ * P_model + (1-λ) * P_kNN
        p_model = self.model_predict(token_id).squeeze()
        p_knn = self.knn_retrieve(token_id)
        return self.lambda_mix * p_model + (1 - self.lambda_mix) * p_knn

# 构建 kNN-LM
vocab_size = 20
knn_lm = SimpleKNNLM(vocab_size, k=5, lambda_mix=0.6)

# 用序列构建数据库
sequences = [
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 6, 7, 8],
    [9, 10, 11, 12, 13],
    [0, 1, 14, 15, 16],
]
knn_lm.build_database(sequences)

print(f"kNN 数据库: {knn_lm.db_keys.shape[0]} 条记录")
print(f"词表大小: {vocab_size}")

# 对比: 纯模型 vs kNN vs 混合
token_id = 0
p_model = knn_lm.model_predict(token_id).squeeze()
p_knn = knn_lm.knn_retrieve(token_id)
p_mixed = knn_lm.predict(token_id)

print(f"\\n给定 token={token_id}, 预测下一个 token:")
print(f"  纯模型 top3: {p_model.topk(3).indices.tolist()}")
print(f"  纯 kNN top3: {p_knn.topk(3).indices.tolist()}")
print(f"  混合   top3: {p_mixed.topk(3).indices.tolist()}")
print("kNN 检索到的 (1,2,14) 正是数据库中 token 0 后面出现的。")""")

# ============================================================
md("""## 3. RETRO：检索增强 Transformer

### 3.1 RETRO 的架构

RETRO 在 Transformer 中插入 **cross-attention 层**，让模型关注检索到的邻居：

```
输入: [token1, token2, ..., tokenN]
  ↓
检索: 每个 chunk 检索 k 个邻居
  ↓
Encoder: 编码邻居 → neighbor_embeddings
  ↓
Transformer + CrossAttention:
  h = SelfAttention(h)         # 自注意力
  h = CrossAttention(h, nbrs)  # 关注检索邻居
  ↓
输出: P(next_token)
```

### 3.2 Chunked Cross-Attention

RETRO 把输入分成 **chunk**（如 32 token），每个 chunk 检索邻居：

$$h = h + \\text{CrossAttn}(Q=h, K=\\text{nbrs}, V=\\text{nbrs})$$

> 💡 RETRO 用 7B 参数 + 检索，超过了 175B GPT-3 的效果。
> 检索让小模型有大模型的知识——这是**外挂知识**的力量。""")

code("""class ChunkedCrossAttention(nn.Module):
    # RETRO 的 chunked cross-attention
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, h, neighbors):
        # h: (batch, seq_len, d) - 输入表示
        # neighbors: (batch, n_chunks, k_nbrs, chunk_len, d) - 检索邻居
        B, T, D = h.shape
        Q = self.W_q(h)
        # 简化: 把所有邻居展平当 K, V
        nbr_flat = neighbors.reshape(B, -1, D)
        K = self.W_k(nbr_flat)
        V = self.W_v(nbr_flat)

        # 多头
        Q = Q.reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.reshape(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.reshape(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)
        return self.W_o(out)

class SimpleRETRO(nn.Module):
    # 简化 RETRO: 自注意力 + 检索 cross-attention
    def __init__(self, vocab_size=20, d_model=32, n_heads=4, chunk_size=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # 检索 cross-attention
        self.cross_attn = ChunkedCrossAttention(d_model, n_heads)
        # 邻居编码器 (简化: 用嵌入)
        self.nbr_embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, neighbors):
        # x: (B, T) token ids
        # neighbors: (B, n_nbrs, chunk_size) 检索到的邻居 token ids
        B, T = x.shape
        h = self.embed(x) + self.pos[:, :T]

        # 自注意力
        h2, _ = self.self_attn(h, h, h)
        h = h + h2

        # 检索 cross-attention
        nbr_h = self.nbr_embed(neighbors)  # (B, n_nbrs, chunk_size, d)
        # 加一维当 n_chunks=1
        nbr_h = nbr_h.unsqueeze(1)  # (B, 1, n_nbrs, chunk_size, d)
        h2 = self.cross_attn(h, nbr_h)
        h = h + h2

        return self.head(h)

# 演示 RETRO
retro = SimpleRETRO(vocab_size=20, d_model=32, chunk_size=4)
x = torch.randint(0, 20, (2, 8))  # batch=2, seq_len=8
neighbors = torch.randint(0, 20, (2, 3, 4))  # 3个邻居, 每个长度4
output = retro(x, neighbors)
print(f"RETRO 输入: {x.shape}, 邻居: {neighbors.shape}")
print(f"RETRO 输出: {output.shape}")
print("RETRO = 自注意力 + 检索cross-attention → 融合外部知识。")""")

# ============================================================
md("""## 4. 检索 vs 长上下文

### 4.1 两种利用外部知识的方式

```
长上下文:  把所有可能相关的放进 context window → 让注意力自己找
检索:      先检索最相关的 → 只把 top-k 放进 context
```

### 4.2 对比

| | 长上下文 | 检索增强 |
|---|---------|---------|
| **复杂度** | O(T²) 注意力 | O(k) 检索 + O(k²) 注意力 |
| **知识量** | 受 context 限制 | 数据库可以无限大 |
| **精确度** | 注意力可能分散 | 检索精确聚焦 |
| **延迟** | 长 context 推理慢 | 检索快, context 短 |

> 💡 2024 年趋势：长上下文（1M token）在挑战检索，但检索在**精确知识**上仍然更强。
> 两者不矛盾——RAG 就是"先检索再放长上下文"。""")

code("""# 模拟: 检索 vs 长上下文的效果对比
np.random.seed(42)

# 模拟: 数据库有 10000 条知识, 问题需要其中 1 条
db_size = 10000
knowledge = np.random.randn(db_size, 64)  # 10000条知识的嵌入

# 检索: 找最相关的 k 条
def retrieval_approach(query, k=10):
    # 检索 top-k, 然后只处理 k 条
    sims = knowledge @ query
    topk_idx = np.argsort(sims)[-k:]
    # 简化: 处理 k 条的复杂度 = O(k²)
    return k * k, topk_idx

# 长上下文: 把 N 条都放进 context
def long_context_approach(query, n_context=1000):
    # 处理 N 条的复杂度 = O(N²)
    return n_context * n_context

# 对比不同数据库大小
db_sizes = [100, 500, 1000, 5000, 10000, 50000]
retrieval_costs = []
long_ctx_costs = []

for ds in db_sizes:
    query = np.random.randn(64)
    # 检索: 固定 k=10, 复杂度不变
    retrieval_costs.append(10 * 10)  # O(k²)
    # 长上下文: 把所有都放进去
    long_ctx_costs.append(ds * ds)  # O(N²)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(db_sizes, retrieval_costs, 'b-o', linewidth=2.5, markersize=8, label='检索 (O(k²), k=10)')
ax.plot(db_sizes, long_ctx_costs, 'r-s', linewidth=2.5, markersize=8, label='长上下文 (O(N²))')
ax.set_xlabel('数据库大小 N')
ax.set_ylabel('计算复杂度')
ax.set_title('检索 vs 长上下文: 复杂度对比')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('notebooks/fig_retrieval_vs_longctx.png', bbox_inches='tight')
plt.show()
print("检索复杂度恒定 O(k²); 长上下文 O(N²) 随数据库爆炸。")""")

# ============================================================
md("""## 5. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 检索增强训练的概念 | ✅ |
| kNN-LM: k近邻语言模型 | ✅ |
| RETRO: 检索增强 Transformer | ✅ |
| Chunked cross-attention | ✅ |
| 检索 vs 长上下文对比 | ✅ |

### 核心 takeaway

> **检索让小模型有大知识**——RETRO 7B 超过 GPT-3 175B。
> 检索增强训练在训练时就学会利用外部知识，比推理时 RAG 更深度融合。
> 检索 vs 长上下文不矛盾——RAG 是两者的结合。

### 🔗 下一章预告

**`18_non_gradient_optimization.ipynb`** — 神经进化/ES/NAS 非梯度优化

---

> 💬 **写在最后**：参数不是唯一的知识存储——外挂数据库让模型更高效。
> 检索增强训练是"小模型大知识"的关键路径。""")

# ============================================================
output_path = "notebooks/17_retrieval_augmented_training.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")