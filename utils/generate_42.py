# 生成 42_pruning_moe.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 42 — 剪枝、稀疏化与 MoE 效率

> 删掉不重要的权重（剪枝），只激活部分专家（MoE）——让模型更高效。

## 本章你将掌握

1. **剪枝**：结构化/非结构化
2. **稀疏化**：L1 正则 + Lottery Ticket
3. **MoE 效率**：稀疏激活
4. **组合优化**""")

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

md("""## 1. 剪枝

### 1.1 非结构化剪枝

把**小的权重**设为 0：

```
权重重要性: |w| (绝对值)
阈值: 删掉最小的 p% 权重

→ 稀疏矩阵 → 理论上减少计算
→ 但需要稀疏硬件支持
```

### 1.2 结构化剪枝

删掉**整个通道/层**：

```
通道重要性: 通道权重的范数
删掉不重要的通道

→ 直接减少维度 → 任何硬件都加速
→ 但可能删掉重要信息
```

> 💡 非结构化剪枝压缩比高但需特殊硬件；结构化剪枝通用但压缩比低。""")

code("""# 剪枝实现
def unstructured_prune(weight, sparsity=0.5):
    # 非结构化: 删小权重
    threshold = torch.quantile(weight.abs().flatten(), sparsity)
    mask = weight.abs() > threshold
    return weight * mask, mask

def structured_prune(weight, sparsity=0.5):
    # 结构化: 删整通道
    channel_importance = weight.abs().sum(dim=1)  # 按行求和
    n_keep = int(weight.shape[0] * (1 - sparsity))
    keep_indices = channel_importance.argsort(descending=True)[:n_keep]
    pruned = weight[keep_indices]
    return pruned, keep_indices

# 演示
w = torch.randn(128, 64) * 0.1

w_unstruct, mask = unstructured_prune(w, sparsity=0.5)
w_struct, indices = structured_prune(w, sparsity=0.5)

print(f"原始: {w.shape}, 非零: {(w != 0).sum().item()}")
print(f"非结构化剪枝 50%: {w_unstruct.shape}, 非零: {(w_unstruct != 0).sum().item()}, 稀疏: {1 - (w_unstruct != 0).sum().item() / w.numel():.1%}")
print(f"结构化剪枝 50%: {w_struct.shape} (删了一半通道)")
print("非结构化: 稀疏矩阵; 结构化: 直接变小。")""")

md("""## 2. Lottery Ticket Hypothesis

### 2.1 假说

**彩票假说**：密集网络中存在一个稀疏子网络（"中奖彩票"），从头训练就能达到原网络性能。

```
1. 训练完整网络 → 得到权重 W
2. 剪枝 → 得到掩码 M (保留 p%)
3. 重置: 用初始权重 W0 * M
4. 从头训练 → 接近原网络性能!
```

### 2.2 意义

- 说明**稀疏网络本身就够**
- 不需要训练大网络再剪枝
- 但找到"中奖彩票"不容易

> 💡 彩票假说揭示：大网络中藏着小网络，从头训练就能好。""")

md("""## 3. MoE 效率

### 3.1 MoE 的稀疏激活

```
稠密模型: 每个 token 过所有参数 → 计算量 = N
MoE 模型:  每个 token 只过 top-k 专家 → 计算量 = k/N * 总参数

例: 8 专家 top-2 → 只用 25% 参数 → 2x 加速
```

### 3.2 MoE 的挑战

```
1. 显存: 所有专家都要加载 (即使不用)
   → 8 专家 70B → 需要存 70B 参数

2. 通信: 专家分散在不同 GPU → all-to-all
   → 通信开销

3. 负载均衡: 有些专家被频繁选, 有些闲置
   → 辅助损失鼓励均衡
```

### 3.3 解决方案

```
1. 专家并行: 每台 GPU 放几个专家
2. 量化 MoE: 量化不活跃专家
3. 专家缓存: 只加载活跃专家到 GPU
```

> 💡 MoE 用稀疏激活换更多参数——但显存和通信是挑战。
# 专家并行 + 量化是当前主流方案。""")

code("""# MoE 效率模拟
def moe_efficiency(n_experts=8, top_k=2, n_layers=32, d_model=4096):
    # 稠密模型
    dense_params = n_layers * 4 * d_model ** 2  # 4*d^2 per layer (FFN)
    dense_flops = dense_params  # 每个 token 的 FLOPS

    # MoE 模型
    moe_total_params = n_layers * n_experts * 4 * d_model ** 2  # 总参数
    moe_active_params = n_layers * top_k * 4 * d_model ** 2  # 激活参数
    moe_flops = moe_active_params  # 每个 token 的 FLOPS

    return {
        'dense_params': dense_params,
        'moe_total': moe_total_params,
        'moe_active': moe_active_params,
        'speedup': dense_flops / moe_flops,
        'param_ratio': moe_total_params / dense_params,
    }

# 不同配置
configs = [
    (8, 2, "Mixtral 8x7B"),
    (64, 2, "DeepSeek 64专家"),
    (128, 2, "GShard 128专家"),
    (256, 8, "MoE 256/8"),
]

print(f"{'配置':>20s} {'总参数比':>10s} {'激活参数比':>10s} {'加速比':>10s}")
for n, k, name in configs:
    r = moe_efficiency(n_experts=n, top_k=k)
    print(f"{name:>20s} {r['moe_total']/r['dense_params']:10.1f}x {r['moe_active']/r['dense_params']:10.1f}x {r['speedup']:10.1f}x")

print("\\nMoE: 总参数多, 激活参数少 → 更多容量, 更快推理。")""")

code("""# 可视化 MoE 效率
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 稀疏激活
ax = axes[0]
n_experts = 8
top_k = 2
expert_usage = np.random.dirichlet(np.ones(n_experts) * 2)
bars = ax.bar(range(n_experts), expert_usage, color='steelblue', alpha=0.8)
# 标记 top-k
top_indices = np.argsort(expert_usage)[-top_k:]
for i in top_indices:
    bars[i].set_color('coral')
ax.set_xlabel('专家'); ax.set_ylabel('使用率')
ax.set_title(f'MoE 激活: top-{top_k} of {n_experts} (红色=激活)')

# 剪枝 vs MoE 对比
ax = axes[1]
methods = ['稠密', '剪枝\\n50%', '量化\\nINT4', 'MoE\\n8x/2', 'MoE+量化']
speeds = [1, 1.5, 4, 2, 8]
qualities = [100, 95, 98, 100, 98]

x = np.arange(len(methods))
ax.bar(x - 0.15, speeds, 0.3, label='加速比', color='steelblue', alpha=0.8)
ax.bar(x + 0.15, [q/10 for q in qualities], 0.3, label='质量(%)', color='coral', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
ax.set_title('压缩方法对比'); ax.legend()

plt.tight_layout()
plt.savefig('notebooks/fig_pruning_moe.png', bbox_inches='tight')
plt.show()
print("MoE+量化: 8x 加速, 质量几乎不降——最强组合。")""")

md("""## 4. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 结构化/非结构化剪枝 | ✅ |
| 彩票假说 | ✅ |
| MoE 稀疏激活 | ✅ |
| MoE 效率与挑战 | ✅ |

### 核心 takeaway
> **剪枝删冗余，MoE 用稀疏激活换容量**。
# MoE + 量化是当前最高效的方案——更多参数，更快推理。

### 🔗 下一章
**`43_paged_attention_serving.ipynb`** — PagedAttention + serving

---

> 💬 **板块七进行中。**""")

output_path = "notebooks/42_pruning_moe.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")