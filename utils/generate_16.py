"""生成 16_model_merging.ipynb 的脚本"""
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
md("""# 16 — 模型合并：TIES / DARE / SLERP

> 微调了多个模型（代码模型、数学模型、聊天模型），能不能**合并**成一个全能模型？
> 模型合并（Model Merging）说：可以！而且不需要额外训练。
> 这是 2024 年的热门方向——用算术操作合并权重。

## 本章你将掌握

1. **简单插值**：线性加权平均
2. **SLERP**：球面线性插值
3. **TIES**：修剪 + 电符号合并
4. **DARE**：丢弃 + 重缩放
5. **实验对比**：不同合并方法的效果""")

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
md("""## 1. 模型合并的直觉

### 1.1 为什么能合并？

微调同一个基座模型得到的多个模型，权重在**同一空间**中。
它们的差异 $\\Delta W_i = W_i - W_{base}$ 代表各自学到的能力。

```
基座模型 W_base
  ├── 代码模型: W_code = W_base + ΔW_code
  ├── 数学模型: W_math = W_base + ΔW_math
  └── 聊天模型: W_chat = W_base + ΔW_chat

合并: W_merged = W_base + ΔW_code + ΔW_math + ΔW_chat  (加法合并)
```

### 1.2 挑战

直接相加有问题：
- **干扰**：$\\Delta W_{code}$ 可能干扰数学能力
- **尺度**：多个 $\\Delta W$ 相加可能太大
- **方向冲突**：两个模型的更新方向可能矛盾

> 💡 TIES/DARE 就是解决这些干扰问题的智能合并方法。""")

# ============================================================
md("""## 2. 简单方法：线性插值

### 2.1 线性加权平均

$$W_{merged} = \\alpha W_A + (1-\\alpha) W_B$$

$\\alpha = 0.5$ 就是简单平均。问题：在高维空间中，线性平均可能不在好的方向上。

### 2.2 SLERP：球面线性插值

SLERP 在**球面**上插值，保持权重的范数：

$$\\text{SLERP}(W_A, W_B, t) = \\frac{\\sin((1-t)\\theta)}{\\sin\\theta} W_A + \\frac{\\sin(t\\theta)}{\\sin\\theta} W_B$$

其中 $\\theta$ 是 $W_A$ 和 $W_B$ 的夹角。

```
线性插值:  W_A ----●---- W_B     (走直线, 可能范数变小)
SLERP:    W_A ----●---- W_B     (走大圆, 保持范数)
```

> 💡 SLERP 在 3D 旋转插值中很常用，模型合并中它保持权重能量。""")

code("""def linear_merge(w_a, w_b, alpha=0.5):
    \"\"\"线性插值\"\"\"
    return alpha * w_a + (1 - alpha) * w_b

def slerp_merge(w_a, w_b, t=0.5):
    \"\"\"球面线性插值\"\"\"
    # 把权重展平成向量
    flat_a = w_a.flatten()
    flat_b = w_b.flatten()

    # 计算夹角
    dot = torch.dot(flat_a, flat_b)
    norm_a = flat_a.norm()
    norm_b = flat_b.norm()
    cos_theta = dot / (norm_a * norm_b + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.arccos(cos_theta)

    if theta.abs() < 1e-6:
        # 几乎同向, 退化为线性
        result = linear_merge(w_a, w_b, t)
    else:
        # SLERP
        sin_theta = torch.sin(theta)
        coeff_a = torch.sin((1 - t) * theta) / sin_theta
        coeff_b = torch.sin(t * theta) / sin_theta
        result_flat = coeff_a * flat_a + coeff_b * flat_b
        result = result_flat.reshape(w_a.shape)

    return result

# 演示
torch.manual_seed(42)
w_a = torch.randn(100)
w_b = torch.randn(100)

alphas = np.linspace(0, 1, 50)
norms_linear = [linear_merge(w_a, w_b, a).norm().item() for a in alphas]
norms_slerp = [slerp_merge(w_a, w_b, a).norm().item() for a in alphas]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(alphas, norms_linear, 'b-', linewidth=2.5, label='线性插值')
ax.plot(alphas, norms_slerp, 'r-', linewidth=2.5, label='SLERP')
ax.set_xlabel('插值参数 α')
ax.set_ylabel('权重范数')
ax.set_title('线性插值 vs SLERP: 范数保持')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_slerp.png', bbox_inches='tight')
plt.show()
print("SLERP 保持范数; 线性插值在中间点范数下降。")""")

# ============================================================
md("""## 3. TIES：修剪 + 电符号合并

### 3.1 TIES 三步

TIES (Trim, Elect Sign, Merge) 解决多模型合并的干扰问题：

```
Step 1 Trim:      修剪每个 ΔW 的小值 (只保留 top-k%)
Step 2 Elect Sign: 对每个参数, 用多数投票决定符号
Step 3 Merge:     按投票符号合并, 冲突的取平均
```

### 3.2 为什么有效？

- **Trim**：去掉噪声（小值多半是噪声）
- **Elect Sign**：解决方向冲突（两个模型一个要正一个要负，投票决定）
- **Merge**：只合并不冲突的参数

> 💡 TIES 论文显示：合并 7 个不同任务模型，效果接近多任务训练。""")

code("""def ties_merge(base_weight, task_weights, density=0.5):
    \"\"\"TIES: Trim + Elect Sign + Merge\"\"\"
    # 1. 计算每个任务的 ΔW
    deltas = [w - base_weight for w in task_weights]

    # 2. Trim: 保留每个 ΔW 的 top density% (按绝对值)
    trimmed = []
    for delta in deltas:
        flat = delta.flatten().abs()
        threshold = torch.quantile(flat, 1 - density)
        mask = delta.abs() >= threshold
        trimmed.append(delta * mask)

    # 3. Elect Sign: 多数投票决定符号
    sign_sum = sum(torch.sign(t) for t in trimmed)
    elected_sign = torch.sign(sign_sum)

    # 4. Merge: 只保留与选举符号一致的参数
    merged_delta = torch.zeros_like(base_weight)
    count = torch.zeros_like(base_weight)
    for t in trimmed:
        consistent = torch.sign(t) == elected_sign
        merged_delta += t * consistent
        count += consistent.float()

    # 避免除零
    count = torch.clamp(count, min=1)
    merged_delta = merged_delta / count

    return base_weight + merged_delta

# 模拟: 基座模型 + 3个任务模型
torch.manual_seed(42)
base = torch.randn(50, 50) * 0.1
# 3个任务各有不同的稀疏更新
task1 = base.clone(); task1[10:20, 10:20] += torch.randn(10, 10) * 0.5  # 任务1更新左上
task2 = base.clone(); task2[30:40, 30:40] += torch.randn(10, 10) * 0.5  # 任务2更新右下
task3 = base.clone(); task3[20:30, 20:30] += torch.randn(10, 10) * 0.5  # 任务3更新中间

# 简单平均
naive_merge = (task1 + task2 + task3) / 3

# TIES
ties_result = ties_merge(base, [task1, task2, task3], density=0.3)

# 计算合并后保留了多少任务信息
def task_performance(merged, task_weight, base_weight):
    \"\"\"简化: 合并模型在某任务上的性能 = 保留了该任务更新的多少\"\"\"
    delta_task = task_weight - base_weight
    delta_merged = merged - base_weight
    # 相关性
    return torch.dot(delta_task.flatten(), delta_merged.flatten()).item() / (delta_task.norm().item() * delta_merged.norm().item() + 1e-8)

print("合并后各任务性能保留 (相关性):")
for i, task in enumerate([task1, task2, task3], 1):
    perf_naive = task_performance(naive_merge, task, base)
    perf_ties = task_performance(ties_result, task, base)
    print(f"  任务{i}: 简单平均={perf_naive:.3f}, TIES={perf_ties:.3f}")

print("\\nTIES 通过修剪和符号投票减少任务间干扰。")""")

# ============================================================
md("""## 4. DARE：丢弃 + 重缩放

### 4.1 DARE 的洞察

微调的 $\\Delta W$ 大部分参数是**冗余的**——可以随机丢弃 90% 而几乎不影响性能！

```
DARE:
  1. 随机丢弃 ΔW 的 p% 参数 (置零)
  2. 重缩放: 剩余参数 × 1/(1-p) 补偿
  3. 合并多个 DARE 后的 ΔW
```

### 4.2 DARE vs TIES

| | TIES | DARE |
|---|------|------|
| **修剪** | 按绝对值 top-k | 随机丢弃 |
| **冲突处理** | 符号投票 | 直接相加 |
| **重缩放** | 无 | $\\frac{1}{1-p}$ |

> 💡 DARE 更简单——随机丢弃就能消除大部分干扰，不需要复杂的符号投票。""")

code("""def dare_merge(base_weight, task_weights, drop_rate=0.9):
    \"\"\"DARE: Drop And Rescale\"\"\"
    # 1. 计算每个任务的 ΔW
    deltas = [w - base_weight for w in task_weights]

    # 2. DARE: 随机丢弃 + 重缩放
    rescaled = 1.0 / (1.0 - drop_rate)
    dare_deltas = []
    for delta in deltas:
        mask = (torch.rand_like(delta) > drop_rate).float()
        dare_deltas.append(delta * mask * rescaled)

    # 3. 直接相加合并
    merged_delta = sum(dare_deltas) / len(dare_deltas)

    return base_weight + merged_delta

# 对比不同合并方法
methods = {
    '简单平均': (task1 + task2 + task3) / 3,
    'TIES (density=0.3)': ties_merge(base, [task1, task2, task3], density=0.3),
    'DARE (drop=0.9)': dare_merge(base, [task1, task2, task3], drop_rate=0.9),
    'TIES (density=0.5)': ties_merge(base, [task1, task2, task3], density=0.5),
}

print("不同合并方法对比 (各任务性能保留):")
print(f"{'方法':20s} | {'任务1':8s} | {'任务2':8s} | {'任务3':8s} | {'平均':8s}")
print("-" * 60)
for name, merged in methods.items():
    perfs = [task_performance(merged, task, base) for task in [task1, task2, task3]]
    avg = np.mean(perfs)
    print(f"{name:20s} | {perfs[0]:8.3f} | {perfs[1]:8.3f} | {perfs[2]:8.3f} | {avg:8.3f}")""")

code("""# 可视化合并方法对比
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 各方法对各任务的性能
method_names = list(methods.keys())
task_names = ['任务1', '任务2', '任务3']
x = np.arange(len(task_names))
width = 0.2

for i, name in enumerate(method_names):
    merged = methods[name]
    perfs = [task_performance(merged, task, base) for task in [task1, task2, task3]]
    axes[0].bar(x + i*width, perfs, width, label=name, alpha=0.8)

axes[0].set_xticks(x + width*1.5)
axes[0].set_xticklabels(task_names)
axes[0].set_ylabel('性能保留 (相关性)')
axes[0].set_title('不同合并方法: 各任务性能')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis='y')

# DARE 不同 drop rate
drop_rates = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
avg_perfs = []
for dr in drop_rates:
    merged = dare_merge(base, [task1, task2, task3], drop_rate=dr)
    perfs = [task_performance(merged, task, base) for task in [task1, task2, task3]]
    avg_perfs.append(np.mean(perfs))

axes[1].plot(drop_rates, avg_perfs, 'b-o', linewidth=2.5, markersize=8)
axes[1].set_xlabel('丢弃率 p')
axes[1].set_ylabel('平均性能保留')
axes[1].set_title('DARE: 不同丢弃率的效果')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_model_merging.png', bbox_inches='tight')
plt.show()
print("DARE 即使丢弃90%参数, 性能保留仍然不错——微调权重高度冗余。")""")

# ============================================================
md("""## 5. 实际应用场景

### 5.1 什么时候用模型合并？

| 场景 | 方法 | 说明 |
|------|------|------|
| **多任务合并** | TIES | 合并不同任务的微调模型 |
| **能力增强** | DARE | 合并代码+数学+聊天模型 |
| **平滑过渡** | SLERP | 两个模型间的平滑过渡 |
| **快速实验** | 线性平均 | 最简单的基线 |

### 5.2 成功案例

- **Llama-2-70B 合并**：合并代码+数学模型，接近多任务性能
- **Zephyr-7B**：用 SLERP 合并 SFT + DPO 模型
- **OpenChat-3.5**：用 TIES 合并多个对话模型

> 💡 模型合并的优势：**零额外训练**——只需算术操作合并权重，几秒钟完成。""")

code("""# 模拟完整合并流程
print("模型合并流程示例:")
print("=" * 50)

# 模拟: 从同一个基座模型微调出3个专家
torch.manual_seed(42)
base_model_weight = torch.randn(20, 20) * 0.1

# 3个专家模型
expert_names = ['代码专家', '数学专家', '对话专家']
expert_weights = []
for name in expert_names:
    w = base_model_weight.clone()
    # 随机更新一部分参数 (模拟微调)
    mask = torch.rand(20, 20) > 0.7
    w[mask] += torch.randn(mask.sum().item()) * 0.3
    expert_weights.append(w)
    delta_norm = (w - base_model_weight).norm().item()
    print(f"  {name}: ΔW 范数 = {delta_norm:.3f}")

print()
# 合并
merged_ties = ties_merge(base_model_weight, expert_weights, density=0.5)
merged_dare = dare_merge(base_model_weight, expert_weights, drop_rate=0.9)
merged_naive = sum(expert_weights) / 3

print("合并结果 (与基座的差异范数):")
print(f"  简单平均: {(merged_naive - base_model_weight).norm().item():.3f}")
print(f"  TIES:     {(merged_ties - base_model_weight).norm().item():.3f}")
print(f"  DARE:     {(merged_dare - base_model_weight).norm().item():.3f}")
print("\\n模型合并: 3个专家 → 1个全能模型, 零额外训练！")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 线性插值合并 | ✅ |
| SLERP 球面插值 | ✅ |
| TIES 修剪+符号投票 | ✅ |
| DARE 随机丢弃+重缩放 | ✅ |
| 不同合并方法对比 | ✅ |

### 核心 takeaway

> **模型合并是"免费"的能力组合**——零训练，几秒完成。
> TIES 解决干扰，DARE 利用冗余，SLERP 保持范数。
> 这是多模型部署的成本节约方案。

### 🔗 下一章预告

**`17_retrieval_augmented_training.ipynb`** — RETRO/kNN-LM 检索增强训练

---

> 💬 **写在最后**：模型合并让"一个模型做所有事"变得简单。
> 微调多个专家 → 合并 → 全能模型，这是高效的能力组合策略。""")

# ============================================================
output_path = "notebooks/16_model_merging.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")