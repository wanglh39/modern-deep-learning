# 生成 43_paged_attention_serving.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 43 — PagedAttention 与 LLM Serving

> vLLM 用 PagedAttention 把 KV cache 当虚拟内存管理，吞吐量提升 2-4 倍。
> 连续批处理让 GPU 永远满载。

## 本章你将掌握

1. **PagedAttention**：分页 KV cache
2. **连续批处理**：动态批处理
3. **LLM Serving**：部署优化
4. **vLLM 架构**""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120; plt.rcParams['font.size'] = 11
np.random.seed(42)
print("环境准备完成 ✅")""")

md("""## 1. KV cache 的问题

### 1.1 传统 KV cache

```
每个请求预分配最大长度的 KV cache:
  请求1 (实际 100 token): 预分配 2048 → 浪费 1948
  请求2 (实际 2000 token): 预分配 2048 → 浪费 48

问题:
  1. 内部碎片: 预分配 > 实际使用
  2. 外部碎片: 请求结束后留下空洞
  3. 显存利用率: ~20-40%
```

### 1.2 PagedAttention 的解决方案

**把 KV cache 分成固定大小的块（页）**，按需分配：

```
传统: 请求 → 预分配连续大块
PagedAttention: 请求 → 多个小页 (按需分配)

类比:
  传统 = 每人预分配大房间 (很多空房间)
  PagedAttention = 按需分配小房间 (紧凑)
```

> 💡 PagedAttention 把 OS 的虚拟内存分页思想用到 KV cache——
# 显存利用率从 20% 提升到 80%+。""")

code("""# PagedAttention 模拟
class PagedKVCache:
    def __init__(self, total_blocks=100, block_size=16):
        self.block_size = block_size
        self.total_blocks = total_blocks
        self.free_blocks = list(range(total_blocks))
        self.request_blocks = {}  # request_id -> [block_ids]

    def allocate(self, request_id, n_tokens):
        n_blocks_needed = (n_tokens + self.block_size - 1) // self.block_size
        if n_blocks_needed > len(self.free_blocks):
            return False  # 显存不足
        blocks = [self.free_blocks.pop() for _ in range(n_blocks_needed)]
        self.request_blocks[request_id] = blocks
        return True

    def extend(self, request_id, additional_tokens):
        n_new_blocks = (additional_tokens + self.block_size - 1) // self.block_size
        for _ in range(n_new_blocks):
            if self.free_blocks:
                self.request_blocks[request_id].append(self.free_blocks.pop())

    def free(self, request_id):
        for block in self.request_blocks[request_id]:
            self.free_blocks.append(block)
        del self.request_blocks[request_id]

    def utilization(self):
        used = self.total_blocks - len(self.free_blocks)
        return used / self.total_blocks

# 对比传统 vs PagedAttention
class TraditionalKVCache:
    def __init__(self, total_memory=1600, max_seq_len=2048):
        self.total_memory = total_memory
        self.max_seq_len = max_seq_len
        self.allocated = {}

    def allocate(self, request_id):
        if self.max_seq_len <= self.remaining():
            self.allocated[request_id] = self.max_seq_len
            return True
        return False

    def remaining(self):
        return self.total_memory - sum(self.allocated.values())

    def utilization(self, actual_usage):
        return sum(actual_usage.values()) / self.total_memory

# 模拟
paged = PagedKVCache(total_blocks=100, block_size=16)
traditional = TraditionalKVCache(total_memory=1600, max_seq_len=2048)

# 10 个请求, 实际长度不同
actual_lengths = [100, 2000, 50, 500, 150, 1000, 80, 300, 120, 700]
paged_actual = {}
trad_actual = {}

for i, length in enumerate(actual_lengths):
    paged.allocate(i, length)
    paged_actual[i] = length
    traditional.allocate(i)
    trad_actual[i] = length

print(f"PagedAttention 利用率: {paged.utilization():.1%}")
print(f"传统利用率: {traditional.utilization(trad_actual):.1%}")
print("PagedAttention: 按需分配 → 显存利用率大幅提升。")""")

md("""## 2. 连续批处理

### 2.1 传统批处理

```
批大小 = 4, 每个请求长度不同:
  请求1: ████████ (完成)
  请求2: ████████████████ (还在生成)
  请求3: ████████ (完成)
  请求4: ████████████ (还在生成)

问题: 请求1/3 完成后要等请求2/4 → GPU 空闲
```

### 2.2 连续批处理

```
请求1/3 完成后立即移出, 加入新请求:
  时刻1: [req1, req2, req3, req4]
  时刻2: [req5, req2, req6, req4]  (req1/3 完成, req5/6 加入)

→ GPU 永远满载 → 吞吐量最大化
```

> 💡 连续批处理 = **动态批处理**：完成的请求立即移出，新请求立即加入。
# GPU 利用率从 50% 提升到 90%+。""")

code("""# 连续批处理模拟
def continuous_batching_sim(n_requests=20, max_batch=4):
    import random
    random.seed(42)

    requests = [(i, random.randint(10, 50)) for i in range(n_requests)]
    completed = []
    gpu_active = []
    timeline = []

    time = 0
    while requests or gpu_active:
        # 移出完成的
        still_active = []
        for req_id, remaining, start_time in gpu_active:
            if time - start_time >= remaining:
                completed.append(req_id)
            else:
                still_active.append((req_id, remaining, start_time))
        gpu_active = still_active

        # 加入新的
        while len(gpu_active) < max_batch and requests:
            req_id, length = requests.pop(0)
            gpu_active.append((req_id, length, time))

        timeline.append((time, len(gpu_active), [r[0] for r in gpu_active]))
        time += 1

    return timeline, completed

timeline, completed = continuous_batching_sim()
active_counts = [t[1] for t in timeline]

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(active_counts)), active_counts, color='steelblue', alpha=0.8)
ax.axhline(4, color='red', linestyle='--', label='最大批大小')
ax.set_xlabel('时间步'); ax.set_ylabel('GPU 活跃请求数')
ax.set_title('连续批处理: GPU 永远满载'); ax.legend()
plt.tight_layout()
plt.savefig('notebooks/fig_continuous_batching.png', bbox_inches='tight')
plt.show()
print(f"完成 {len(completed)} 个请求, {len(timeline)} 步")
print(f"平均 GPU 利用率: {np.mean(active_counts)/4:.1%}")""")

md("""## 3. vLLM 架构

### 3.1 整体设计

```
vLLM = PagedAttention + 连续批处理 + 前缀缓存

组件:
  1. PagedAttention: 分页 KV cache
  2. 连续批处理: 动态批
  3. 前缀缓存: 共享公共前缀 (如 system prompt)
  4. 张量并行: 多 GPU 推理
```

### 3.2 性能

```
vLLM vs HuggingFace:
  吞吐量: 2-4x 提升
  显存利用: 80%+ vs 20-40%
  延迟: 相似或更好
```

### 3.3 其他 serving 系统

| 系统 | 特点 |
|------|------|
| **vLLM** | PagedAttention, 高吞吐 |
| **TensorRT-LLM** | NVIDIA, 极致优化 |
| **TGI** | HuggingFace, 易用 |
| **SGLang** | 结构化生成, 快 |
| **Ollama** | 本地部署, 简单 |

> 💡 vLLM 是当前最流行的 LLM serving 系统——
# PagedAttention + 连续批处理 = 2-4x 吞吐量提升。""")

code("""# Serving 系统对比
systems = ['HuggingFace', 'vLLM', 'TensorRT-LLM', 'SGLang', 'TGI']
throughputs = [1.0, 3.5, 4.0, 3.8, 2.0]  # 相对吞吐量

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(systems, throughputs, color=['gray', 'steelblue', 'coral', 'forestgreen', 'purple'], alpha=0.8)
ax.set_ylabel('相对吞吐量 (x)'); ax.set_title('LLM Serving 系统对比')
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
for bar, t in zip(bars, throughputs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{t}x', ha='center')
plt.tight_layout()
plt.savefig('notebooks/fig_serving.png', bbox_inches='tight')
plt.show()
print("vLLM/TensorRT-LLM/SGLang: 3-4x 吞吐量提升。")""")

md("""## 4. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| KV cache 问题 | ✅ |
| PagedAttention | ✅ |
| 连续批处理 | ✅ |
| vLLM 架构 | ✅ |

### 核心 takeaway
> **PagedAttention 把 KV cache 当虚拟内存，连续批处理让 GPU 满载**。
# vLLM = PagedAttention + 连续批处理 → 2-4x 吞吐量。

### 🔗 下一章
**`44_inference_acceleration.ipynb`** — KV cache、FlashAttention、vLLM

---

> 💬 **板块七进行中。**""")

output_path = "notebooks/43_paged_attention_serving.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")