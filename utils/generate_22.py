"""生成 22_long_context.ipynb 的脚本"""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 22 — 长上下文：Ring Attention、位置编码外推

> 从 4K context 到 1M context——长上下文让 LLM 能处理整本书、整个代码库。
> 但 $O(T^2)$ 注意力和位置编码外推是两大挑战。

## 本章你将掌握

1. **长上下文的挑战**
2. **Ring Attention**：跨设备分片注意力
3. **位置编码外推**：YaRN/NTK-aware
4. **KV cache 压缩**""")

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
plt.rcParams['figure.dpi'] = 120; plt.rcParams['font.size'] = 11
np.random.seed(42); torch.manual_seed(42)
print("环境准备完成 ✅")""")

md("""## 1. 长上下文的挑战

### 1.1 三个瓶颈

```
1. 计算瓶颈:  注意力 O(T²) → 1M token 的注意力矩阵 = 1T 个元素
2. 显存瓶颈:  KV cache 随 T 线性增长 → 1M token 的 KV cache = 几十GB
3. 外推瓶颈:  位置编码在训练长度外性能下降
```

### 1.2 当前长上下文能力

| 模型 | Context 长度 | 方法 |
|------|-------------|------|
| GPT-4 | 128K | Sparse attention |
| Claude 3 | 200K | Ring attention |
| Gemini 1.5 | 1M | Ring attention + 压缩 |
| Llama 3 | 128K | RoPE 外推 |

> 💡 1M token ≈ 75万英文单词 ≈ 一套《哈利波特》全集。""")

md("""## 2. Ring Attention：跨设备分片

### 2.1 问题

单个 GPU 放不下 1M token 的 KV cache。Ring Attention 把序列**分片到多个 GPU**：

```
GPU 0: 处理 chunk 0, KV 0
GPU 1: 处理 chunk 1, KV 1
GPU 2: 处理 chunk 2, KV 2
GPU 3: 处理 chunk 3, KV 3

Ring 传递: KV 0 → GPU1 → GPU2 → GPU3 → GPU0 (环形)
每个 GPU 依次看到所有 KV → 计算完整注意力
```

### 2.2 优势

- **线性扩展**：N 个 GPU → N 倍 context
- **通信与计算重叠**：传 KV 的同时算注意力
- **无需改模型**：纯系统优化

> 💡 Ring Attention 让 1M context 在 8 个 GPU 上可行——系统优化的力量。""")

code("""# 模拟 Ring Attention
def ring_attention_sim(n_gpus=4, chunk_size=256):
    # 模拟: 每个GPU处理一个chunk, 环形传递KV
    total_len = n_gpus * chunk_size
    # 每个GPU的计算量: 自己的Q × 所有KV
    compute_per_gpu = chunk_size * total_len  # O(chunk * total)
    # 通信量: 传N-1次KV
    comm_per_gpu = (n_gpus - 1) * chunk_size
    return total_len, compute_per_gpu, comm_per_gpu

# 不同GPU数量的扩展
n_gpus_list = [1, 2, 4, 8, 16]
results = []
for n in n_gpus_list:
    total, compute, comm = ring_attention_sim(n, chunk_size=8192)
    results.append((n, total, compute, comm))

print("Ring Attention 扩展:")
print(f"{'GPU数':>6s} {'总context':>10s} {'每GPU计算':>12s} {'每GPU通信':>10s}")
for n, total, comp, comm in results:
    print(f"{n:6d} {total:10d} {comp:12d} {comm:10d}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(n_gpus_list, [r[1] for r in results], 'b-o', linewidth=2.5, markersize=8)
axes[0].set_xlabel('GPU 数量'); axes[0].set_ylabel('总 context 长度')
axes[0].set_title('Ring Attention: 线性扩展'); axes[0].grid(True, alpha=0.3)

axes[1].plot(n_gpus_list, [r[2] for r in results], 'r-o', linewidth=2.5, markersize=8, label='计算')
axes[1].plot(n_gpus_list, [r[3] for r in results], 'g-s', linewidth=2.5, markersize=8, label='通信')
axes[1].set_xlabel('GPU 数量'); axes[1].set_ylabel('每GPU开销')
axes[1].set_title('Ring Attention: 计算vs通信'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_ring_attention.png', bbox_inches='tight')
plt.show()
print("Ring Attention: context随GPU数线性增长——N个GPU支持N倍长度。")""")

md("""## 3. 位置编码外推

### 3.1 问题

RoPE 在训练长度 $L_{train}$ 内表现好，但超出后性能下降：

```
训练: 0-4096 位置 → 学会了相对位置关系
推理: 0-128K 位置 → 128K 的位置关系没见过 → 性能下降
```

### 3.2 外推方法

| 方法 | 思想 | 效果 |
|------|------|------|
| **线性插值** | 压缩位置到训练范围 | 一般 |
| **NTK-aware** | 调整RoPE基频率 | 好 |
| **YaRN** | 分段缩放 | 很好 |

### 3.3 NTK-aware RoPE

RoPE 的基频率 $\\theta = 10000$。NTK-aware 增大 $\\theta$ 来支持更长位置：

$$\\theta_{new} = \\theta \\cdot s^{d/(d-2)}$$

其中 $s = L_{target} / L_{train}$ 是扩展比例。

> 💡 YaRN 是当前最好的外推方法——Llama 3 用它从 8K 扩展到 128K。""")

code("""# 位置编码外推演示
def rope_freqs(d_model, max_len, base=10000):
    # RoPE 频率
    freqs = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    return angles

# 训练长度 4096, 外推到 131072
d_model = 64
train_len = 4096
target_len = 131072
scale = target_len / train_len

# 原始 RoPE (不外推)
angles_orig = rope_freqs(d_model, target_len, base=10000)

# NTK-aware: 增大 base
ntk_base = 10000 * (scale ** (d_model / (d_model - 2)))
angles_ntk = rope_freqs(d_model, target_len, base=ntk_base)

# 线性插值: 压缩位置
angles_linear = rope_freqs(d_model, target_len, base=10000)
angles_linear[:, :] *= (train_len / target_len)  # 压缩到训练范围

# 可视化: 不同位置的注意力衰减
def attention_decay(angles, query_pos=0):
    # 查询位置对其他位置的注意力 (简化)
    key_angles = angles[query_pos:] - angles[query_pos:query_pos+1]
    # 用cos相似度近似
    decay = torch.cos(key_angles[:, :8]).mean(dim=-1)
    return decay.numpy()

fig, ax = plt.subplots(figsize=(12, 6))
for name, angles, color in [('原始RoPE', angles_orig, 'blue'),
                              ('线性插值', angles_linear, 'green'),
                              ('NTK-aware', angles_ntk, 'red')]:
    decay = attention_decay(angles)
    positions = np.arange(len(decay))
    ax.plot(positions, decay, color=color, linewidth=2, label=name, alpha=0.8)

ax.axvline(train_len, color='gray', linestyle='--', alpha=0.5, label='训练长度边界')
ax.set_xlabel('位置')
ax.set_ylabel('注意力相似度')
ax.set_title('位置编码外推: 训练长度外的行为')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_rope_extrapolation.png', bbox_inches='tight')
plt.show()
print("训练长度内都正常; 超出后原始RoPE退化, NTK-aware保持稳定。")""")

md("""## 4. KV Cache 压缩

### 4.1 KV cache 的显存问题

```
KV cache = 2 (K+V) × n_layers × n_heads × d_head × seq_len × batch
70B 模型, 128K context: ~40GB KV cache
```

### 4.2 压缩方法

| 方法 | 说明 | 压缩率 |
|------|------|--------|
| **量化** | KV cache 用 8bit/4bit | 2-4x |
| **Eviction** | 丢弃不重要的 KV | 任意 |
| **StreamingLLM** | 只保留开头+最近 | 固定大小 |

> 💡 StreamingLLM 的发现：保留 attention sink（开头几个 token）+ 最近的 KV，
> 中间的全丢掉，效果几乎不降——因为中间的 KV 很少被注意到。""")

code("""# KV cache 压缩模拟
def kv_cache_size(seq_len, n_layers=32, n_heads=32, d_head=128, bytes_per_elem=2):
    return 2 * n_layers * n_heads * d_head * seq_len * bytes_per_elem

seq_lens = [4096, 8192, 16384, 32768, 65536, 131072]
cache_sizes = [kv_cache_size(s) / 1e9 for s in seq_lens]  # GB
cache_4bit = [kv_cache_size(s, bytes_per_elem=0.5) / 1e9 for s in seq_lens]
cache_stream = [kv_cache_size(4096) / 1e9] * len(seq_lens)  # 固定大小

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(seq_lens, cache_sizes, 'r-o', linewidth=2.5, markersize=7, label='fp16 (原始)')
ax.plot(seq_lens, cache_4bit, 'b-s', linewidth=2.5, markersize=7, label='4bit 量化')
ax.plot(seq_lens, cache_stream, 'g-^', linewidth=2.5, markersize=7, label='StreamingLLM')
ax.set_xlabel('序列长度'); ax.set_ylabel('KV Cache (GB)')
ax.set_title('KV Cache 压缩: 70B 模型')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_kv_cache.png', bbox_inches='tight')
plt.show()
print("StreamingLLM: KV cache固定大小, 不随序列长度增长——无限context成为可能。")""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 长上下文三大瓶颈 | ✅ |
| Ring Attention 跨设备分片 | ✅ |
| 位置编码外推 (NTK/YaRN) | ✅ |
| KV cache 压缩 | ✅ |

### 核心 takeaway
> **长上下文 = 系统优化 + 位置编码外推**。
> Ring Attention 解决计算, YaRN 解决外推, StreamingLLM 解决显存。
> 1M context 让 LLM 能处理整本书/整个代码库。

### 🔗 下一板块
**`23_clip_blip.ipynb`** — CLIP/BLIP 多模态对齐（进入多模态与具身板块）

---

> 💬 **板块三(推理与推理模型)完结。**""")

output_path = "notebooks/22_long_context.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")