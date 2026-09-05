# 生成 44_inference_acceleration.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 44 — 推理加速：FlashAttention、投机解码

> FlashAttention 让注意力不写回 HBM，投机解码用小模型加速大模型。

## 本章你将掌握

1. **FlashAttention**：IO 感知的注意力
2. **投机解码**：小模型加速大模型
3. **KV cache 复用**：前缀缓存
4. **综合优化**""")

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

md("""## 1. FlashAttention

### 1.1 标准 Attention 的问题

```
标准注意力:
  1. S = Q @ K.T          → 写到 HBM (大矩阵)
  2. P = softmax(S)       → 写到 HBM
  3. O = P @ V            → 写到 HBM

  HBM 读写: O(N²d) → 内存带宽瓶颈
```

### 1.2 FlashAttention 的解决方案

**分块计算，不写中间矩阵到 HBM**：

```
FlashAttention:
  1. 把 Q, K, V 分成小块
  2. 在 SRAM 中算完整注意力 (不写 HBM)
  3. 只写最终结果 O 到 HBM

  HBM 读写: O(N²d² / M) → M 是 SRAM 大小
  → 大幅减少 HBM 访问
```

### 1.3 效果

```
FlashAttention v1: 2-4x 加速
FlashAttention v2: 2x 进一步加速
FlashAttention v3: 利用 H100 异步 → 1.5-2x

→ 几乎免费的加速 (不改变结果)
```

> 💡 FlashAttention 是**IO 优化**而非计算优化——
# 减少内存访问次数，不是减少计算量。结果完全相同。""")

code("""# FlashAttention vs 标准 Attention
def standard_attention(Q, K, V):
    # 标准: 完整矩阵
    S = Q @ K.transpose(-2, -1)
    P = torch.softmax(S, dim=-1)
    O = P @ V
    return O

def flash_attention_sim(Q, K, V, block_size=64):
    # 简化 FlashAttention: 分块计算
    N = Q.shape[1]
    O = torch.zeros_like(Q)
    for i in range(0, N, block_size):
        O_block = torch.zeros_like(Q[:, i:i+block_size])
        for j in range(0, N, block_size):
            Q_block = Q[:, i:i+block_size]
            K_block = K[:, j:j+block_size]
            V_block = V[:, j:j+block_size]
            S_block = Q_block @ K_block.transpose(-2, -1)
            P_block = torch.softmax(S_block, dim=-1)
            O_block += P_block @ V_block
        O[:, i:i+block_size] = O_block
    return O

# 对比
N, d = 256, 64
Q = torch.randn(1, N, d)
K = torch.randn(1, N, d)
V = torch.randn(1, N, d)

import time
start = time.time()
o_standard = standard_attention(Q, K, V)
t_standard = time.time() - start

start = time.time()
o_flash = flash_attention_sim(Q, K, V, block_size=64)
t_flash = time.time() - start

error = (o_standard - o_flash).abs().max().item()
print(f"标准 Attention: {t_standard*1000:.2f}ms")
print(f"FlashAttention: {t_flash*1000:.2f}ms")
print(f"最大误差: {error:.6f} (应该 ≈ 0)")
print("FlashAttention: 结果相同, IO 更少 → 实际中 2-4x 加速。")""")

md("""## 2. 投机解码

### 2.1 思想

用**小模型**快速生成草稿，**大模型**验证：

```
小模型 (快): 生成 k 个 token 草稿
大模型 (准): 一次性验证 k 个 token
  → 接受前 n 个正确的
  → 拒绝第 n+1 个, 大模型生成正确 token

加速: 小模型猜对多 → 一次出 k 个 token
```

### 2.2 流程

```
1. 小模型生成: t1, t2, t3, t4 (草稿)
2. 大模型并行验证: p1, p2, p3, p4
3. 比较:
   t1 == p1? ✅ 接受
   t2 == p2? ✅ 接受
   t3 == p3? ❌ 拒绝, 用 p3 替代
   → 一次出 3 个 token (而不是 1 个)
```

### 2.3 效果

```
小模型 = 7B, 大模型 = 70B
  → 7B 生成快 10x
  → 如果 80% 猜对 → 2-3x 加速
  → 质量完全等于大模型 (无损)
```

> 💡 投机解码是**无损加速**——输出和大模型完全相同。
# 关键是小模型要"猜得准"——可以用同一模型的量化版。""")

code("""# 投机解码模拟
def speculative_decoding_sim(accept_rate=0.8, draft_size=4, n_tokens=100):
    # 模拟: 小模型猜, 大模型验证
    generated = 0
    steps = 0
    while generated < n_tokens:
        # 小模型生成 draft_size 个草稿
        # 大模型验证
        n_accepted = 0
        for _ in range(draft_size):
            if np.random.random() < accept_rate:
                n_accepted += 1
            else:
                break
        # 接受 n_accepted 个 + 1 个大模型修正
        generated += n_accepted + 1
        steps += 1
    return steps

# 不同接受率
rates = [0.5, 0.7, 0.8, 0.9, 0.95]
speedups = []
for rate in rates:
    steps = speculative_decoding_sim(accept_rate=rate, draft_size=4, n_tokens=1000)
    speedup = 1000 / steps  # 相对于纯大模型
    speedups.append(speedup)
    print(f"接受率 {rate:.0%}: {steps} 步 → {speedup:.1f}x 加速")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([f'{r:.0%}' for r in rates], speedups, color='steelblue', alpha=0.8)
ax.set_xlabel('小模型接受率'); ax.set_ylabel('加速比')
ax.set_title('投机解码加速 (无损)')
plt.tight_layout()
plt.savefig('notebooks/fig_speculative.png', bbox_inches='tight')
plt.show()
print("投机解码: 接受率越高 → 加速越多, 质量无损。")""")

md("""## 3. 其他推理优化

### 3.1 KV cache 复用

```
多个请求共享相同前缀 (如 system prompt):
  "You are a helpful assistant. ..." → KV cache 只算一次

vLLM 的前缀缓存: 自动检测共享前缀 → 复用 KV
```

### 3.2 模型编译

```
torch.compile:
  - 算子融合 (多个小算子 → 一个大算子)
  - 减少 kernel 启动开销
  - 10-30% 加速

TensorRT:
  - 更激进的融合
  - 针对硬件优化
```

### 3.3 混合精度推理

```
FP16/BF16: 2x 加速, 几乎无损
INT8: 4x 加速, 轻微损
INT4: 8x 加速, 轻小损
```

> 💡 推理优化是组合拳：FlashAttention + 投机解码 + 量化 + 编译 → 10x+ 加速。

### 🔗 下一章
**`45_distributed_training.ipynb`** — DDP/FSDP/分布式训练

---

> 💬 **板块七进行中。**""")

output_path = "notebooks/44_inference_acceleration.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")