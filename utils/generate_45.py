# 生成 45_distributed_training.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 45 — 分布式训练：DDP/FSDP/流水线并行

> 训练 70B 模型需要多 GPU。DDP 复制模型，FSDP 切分参数，流水线切分层。

## 本章你将掌握

1. **数据并行 (DDP)**：每卡完整模型，不同数据
2. **FSDP**：切分参数，节省显存
3. **流水线并行**：切分层
4. **混合精度**：BF16 训练""")

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

md("""## 1. 数据并行 (DDP)

### 1.1 思想

N 个 GPU，每卡完整模型副本，处理不同数据，梯度 all-reduce 同步。batch_size = N × per_gpu_batch → N 倍吞吐量。

### 1.2 限制

每卡要放完整模型。70B 模型 + 优化器状态 ≈ 560GB → 单卡放不下 → 需要 FSDP。

> 💡 DDP 适合**模型放得下单卡**的场景——简单高效。大模型需要 FSDP。""")

md("""## 2. FSDP：完全分片数据并行

### 2.1 思想

把参数、梯度、优化器状态**分片到各卡**：每卡只存 1/N。前向时 all-gather 收集完整参数，算完丢弃。

### 2.2 优势

70B 模型 8 卡 → 每卡 ~70GB → 可行。PyTorch FSDP, DeepSpeed ZeRO-3 都是这个思想。

> 💡 FSDP 是训练大模型的主流方案。""")

code("""# 分布式训练显存对比
def memory_analysis(model_params_B, n_gpus=8, optimizer='adam'):
    bytes_per_param = 2  # BF16
    optimizer_multiplier = 4 if optimizer == 'adam' else 2
    total_memory = model_params_B * bytes_per_param * (1 + 1 + optimizer_multiplier)
    ddp_per_gpu = total_memory
    fsdp_per_gpu = total_memory / n_gpus
    return ddp_per_gpu, fsdp_per_gpu

models = [7, 13, 70, 175]
print(f"{'模型(B)':>8s} {'DDP(GB)':>10s} {'FSDP-8卡(GB)':>14s} {'可行?':>6s}")
for params in models:
    ddp, fsdp = memory_analysis(params, n_gpus=8)
    feasible = "✅" if fsdp < 80 else "❌"
    print(f"{params:8d} {ddp:10.0f} {fsdp:14.0f} {feasible:>6s}")

print("\\nFSDP: 70B 模型 8 卡可行 (每卡 ~70GB)。")""")

md("""## 3. 流水线并行

### 3.1 思想

把模型**按层切分**到不同 GPU：GPU 0 处理层 1-10，GPU 1 处理层 11-20，...。不同 micro-batch 在不同 GPU 同时计算。

### 3.2 1F1B 调度

流水线填充+排空，减少 bubble。和 FSDP 组合使用 (3D 并行 = 数据 + 流水线 + 张量)。

> 💡 流水线并行适合**层很多**的模型。""")

code("""# 3D 并行可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.set_title('数据并行 (DDP)', fontsize=12)
for i in range(4):
    ax.add_patch(plt.Rectangle((0.1, i*0.2+0.1), 0.8, 0.15, facecolor='lightblue', edgecolor='black'))
    ax.text(0.5, i*0.2+0.175, f'GPU {i}: 完整模型', ha='center', fontsize=9)
ax.text(0.5, 0.05, '不同数据', ha='center', fontsize=9, color='red')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

ax = axes[1]
ax.set_title('FSDP', fontsize=12)
for i in range(4):
    ax.add_patch(plt.Rectangle((0.1, i*0.2+0.1), 0.8, 0.15, facecolor='lightyellow', edgecolor='black'))
    ax.text(0.5, i*0.2+0.175, f'GPU {i}: 1/4 参数', ha='center', fontsize=9)
ax.text(0.5, 0.05, '参数分片', ha='center', fontsize=9, color='red')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

ax = axes[2]
ax.set_title('流水线并行', fontsize=12)
for i in range(4):
    ax.add_patch(plt.Rectangle((i*0.2+0.05, 0.3), 0.15, 0.4, facecolor='lightgreen', edgecolor='black'))
    ax.text(i*0.2+0.125, 0.5, f'GPU {i}\\n层{i*10}-{(i+1)*10}', ha='center', fontsize=8)
ax.text(0.5, 0.15, '层切分', ha='center', fontsize=9, color='red')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

plt.tight_layout()
plt.savefig('notebooks/fig_3d_parallel.png', bbox_inches='tight')
plt.show()
print("3D 并行 = 数据 + 流水线 + 张量 → 训练超大模型。")""")

md("""## 4. 混合精度训练

### 4.1 BF16 vs FP32

- **FP32**: 1符号 + 8指数 + 23尾数 → 精度高
- **BF16**: 1符号 + 8指数 + 7尾数 → 范围同FP32, 精度低
- **FP16**: 1符号 + 5指数 + 10尾数 → 范围小, 易溢出

BF16 是训练首选 (范围大, 不溢出)。前向/反向用 BF16，优化器用 FP32 → 2x 加速, 几乎无损。

> 💡 BF16 混合精度是训练标配——2x 加速几乎免费。A100/H100 原生支持。

### 🔗 下一章
**`46_edge_deployment.ipynb`** — WebGPU、本地部署

---

> 💬 **板块七进行中。**""")

output_path = "notebooks/45_distributed_training.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")
