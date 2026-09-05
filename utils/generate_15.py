"""生成 15_training_techniques.ipynb 的脚本"""
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
md("""# 15 — 训练技巧：学习率调度、梯度累积、裁剪

> 好的训练技巧能让同样的模型效果提升 10-20%。
> 本章覆盖现代 LLM 训练的核心技巧：学习率调度、梯度累积、梯度裁剪、混合精度。

## 本章你将掌握

1. **学习率调度**：warmup + cosine decay
2. **梯度累积**：小显存模拟大 batch
3. **梯度裁剪**：防止梯度爆炸
4. **混合精度**：fp16/bf16 加速训练
5. **综合应用**：完整训练循环""")

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
md("""## 1. 学习率调度

### 1.1 为什么需要调度？

学习率是最重要的超参数：
- **太大** → 震荡不收敛
- **太小** → 收敛太慢
- **刚好** → 快速稳定收敛

现代训练用**两阶段**调度：

```
warmup:  lr 从 0 线性升到 max_lr    (前 10% 步骤)
cosine:  lr 从 max_lr 余弦降到 0    (后 90% 步骤)
```

### 1.2 Warmup 的作用

训练初期参数随机，梯度方向噪声大。Warmup 让学习率**慢慢起步**，避免一开始就走偏。

### 1.3 Cosine Decay 的作用

训练后期需要**精细调整**，cosine decay 让学习率平滑下降到接近 0。

$$lr_t = \\frac{1}{2}(1 + \\cos(\\pi t / T)) \\cdot max\\_lr$$""")

code("""def get_lr_warmup_cosine(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    \"\"\"Warmup + Cosine Decay 学习率调度\"\"\"
    if step < warmup_steps:
        # 线性 warmup
        return max_lr * step / warmup_steps
    else:
        # cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

def get_lr_constant(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    \"\"\"恒定学习率\"\"\"
    return max_lr

def get_lr_linear(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    \"\"\"Warmup + 线性衰减\"\"\"
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr - (max_lr - min_lr) * progress

def get_lr_step(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    \"\"\"Warmup + 阶梯衰减\"\"\"
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        if progress < 0.33: return max_lr
        elif progress < 0.66: return max_lr * 0.1
        else: return max_lr * 0.01

# 可视化不同调度策略
total_steps = 1000
warmup_steps = 100
max_lr = 3e-4

steps = np.arange(total_steps)
schedules = {
    'Warmup+Cosine': get_lr_warmup_cosine,
    '恒定': get_lr_constant,
    'Warmup+线性': get_lr_linear,
    'Warmup+阶梯': get_lr_step,
}

fig, ax = plt.subplots(figsize=(11, 6))
for name, fn in schedules.items():
    lrs = [fn(s, total_steps, warmup_steps, max_lr) for s in steps]
    ax.plot(steps, lrs, linewidth=2.5, label=name)

ax.axvline(warmup_steps, color='gray', linestyle='--', alpha=0.5, label='warmup结束')
ax.set_xlabel('训练步数')
ax.set_ylabel('学习率')
ax.set_title('学习率调度策略对比')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_lr_schedule.png', bbox_inches='tight')
plt.show()
print("Warmup+Cosine 是现代 LLM 训练的标准选择——平稳起步, 平滑收尾。")""")

# ============================================================
md("""## 2. 梯度累积：小显存模拟大 batch

### 2.1 问题

大 batch 训练更稳定，但显存不够。梯度累积的思路：

```
真实大 batch (256):  loss = loss(batch_256)      → 需要大显存
梯度累积 (8次×32):   loss = (loss(b1)+...+loss(b8)) / 8  → 小显存, 等效大batch
```

### 2.2 实现

```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # 除以累积步数
    loss.backward()                            # 梯度累积
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()                       # 累积够了才更新
        optimizer.zero_grad()                  # 清零梯度
```

> 💡 梯度累积让 8GB 显存的 GPU 也能模拟 256 的 batch size——等效于 64GB 显存。""")

code("""# 梯度累积演示
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)

# 生成数据
n_samples = 256
X = torch.randn(n_samples, 10)
y = (X[:, 0] + X[:, 1] > 0).float()

# 对比: 大batch vs 梯度累积
def train_big_batch(model, X, y, batch_size=256, epochs=50, lr=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        idx = torch.randperm(n_samples)[:batch_size]
        loss = F.mse_loss(model(X[idx]), y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def train_gradient_accumulation(model, X, y, micro_batch=32, accum_steps=8, epochs=50, lr=1e-2):
    \"\"\"用 micro_batch=32, 累积8步 → 等效 batch=256\"\"\"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        idx = torch.randperm(n_samples)[:micro_batch * accum_steps]
        epoch_loss = 0
        optimizer.zero_grad()
        for i in range(accum_steps):
            micro_idx = idx[i*micro_batch:(i+1)*micro_batch]
            loss = F.mse_loss(model(X[micro_idx]), y[micro_idx]) / accum_steps
            loss.backward()
            epoch_loss += loss.item() * accum_steps
        optimizer.step()  # 累积够了才更新
        losses.append(epoch_loss / accum_steps)
    return losses

torch.manual_seed(42)
model_big = SimpleModel()
losses_big = train_big_batch(model_big, X, y)

torch.manual_seed(42)
model_accum = SimpleModel()
losses_accum = train_gradient_accumulation(model_accum, X, y)

print(f"大batch (256) 最终损失: {losses_big[-1]:.4f}")
print(f"梯度累积 (8×32) 最终损失: {losses_accum[-1]:.4f}")
print("两者应该接近——梯度累积等效于大batch训练。")""")

code("""# 可视化梯度累积对比
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses_big, 'b-', linewidth=2, label='大batch (256)')
ax.plot(losses_accum, 'r--', linewidth=2, label='梯度累积 (8×32)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('梯度累积 vs 大batch: 等效训练')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_grad_accumulation.png', bbox_inches='tight')
plt.show()
print("梯度累积用小显存模拟大batch——曲线几乎重合。")""")

# ============================================================
md("""## 3. 梯度裁剪：防止梯度爆炸

### 3.1 梯度爆炸的问题

训练中梯度可能突然变大，导致参数更新过大，模型崩溃：

```
正常:  grad = 0.1  →  w = w - lr * 0.1  ✅
爆炸:  grad = 100  →  w = w - lr * 100  ❌ (w 飞了)
```

### 3.2 梯度裁剪

把梯度范数限制在阈值内：

$$\\text{clip}(g, c) = \\min\\left(1, \\frac{c}{\\|g\\|}\\right) \\cdot g$$

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```""")

code("""# 梯度裁剪演示
torch.manual_seed(42)

# 一个容易梯度爆炸的模型
class UnstableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x)) * 3  # 放大, 容易梯度爆炸
        return x.sum(dim=-1)

X_train = torch.randn(32, 10)
y_train = torch.randn(32)

# 不裁剪
torch.manual_seed(42)
model_no_clip = UnstableModel()
optimizer = torch.optim.Adam(model_no_clip.parameters(), lr=1e-2)
losses_no_clip = []
grad_norms_no_clip = []

for _ in range(100):
    loss = F.mse_loss(model_no_clip(X_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm()**2 for p in model_no_clip.parameters() if p.grad is not None)**0.5
    grad_norms_no_clip.append(grad_norm.item())
    optimizer.step()
    losses_no_clip.append(loss.item())

# 有裁剪
torch.manual_seed(42)
model_clip = UnstableModel()
optimizer = torch.optim.Adam(model_clip.parameters(), lr=1e-2)
losses_clip = []
grad_norms_clip = []

for _ in range(100):
    loss = F.mse_loss(model_clip(X_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪!
    torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm=1.0)
    grad_norm = sum(p.grad.norm()**2 for p in model_clip.parameters() if p.grad is not None)**0.5
    grad_norms_clip.append(grad_norm.item())
    optimizer.step()
    losses_clip.append(loss.item())

print(f"不裁剪: 最终损失={losses_no_clip[-1]:.2f}, 最大梯度范数={max(grad_norms_no_clip):.1f}")
print(f"有裁剪: 最终损失={losses_clip[-1]:.2f}, 最大梯度范数={max(grad_norms_clip):.1f}")""")

code("""# 可视化梯度裁剪效果
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(losses_no_clip, 'r-', linewidth=2, label='不裁剪')
axes[0].plot(losses_clip, 'b-', linewidth=2, label='梯度裁剪 (max_norm=1.0)')
axes[0].set_xlabel('Step'); axes[0].set_ylabel('Loss')
axes[0].set_title('训练损失: 梯度裁剪防爆炸'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(grad_norms_no_clip, 'r-', linewidth=2, label='不裁剪')
axes[1].plot(grad_norms_clip, 'b-', linewidth=2, label='裁剪')
axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='裁剪阈值')
axes[1].set_xlabel('Step'); axes[1].set_ylabel('梯度范数')
axes[1].set_title('梯度范数: 裁剪限制在阈值内'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_grad_clipping.png', bbox_inches='tight')
plt.show()
print("梯度裁剪把范数限制在阈值内——防止梯度爆炸导致训练崩溃。")""")

# ============================================================
md("""## 4. 混合精度训练

### 4.1 fp32 vs fp16 vs bf16

| 精度 | 字节 | 范围 | 精度 | 速度 |
|------|------|------|------|------|
| **fp32** | 4 | 大 | 高 | 基准 |
| **fp16** | 2 | 小 | 中 | 2x |
| **bf16** | 2 | 大 | 低 | 2x |

### 4.2 自动混合精度 (AMP)

用 fp16 计算前向/反向（快），用 fp32 更新参数（精确）：

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(x)       # fp16 前向
scaler.scale(loss).backward()  # fp16 反向
scaler.step(optimizer)   # fp32 更新
```

> 💡 bf16 是现代 LLM 的首选——和 fp32 一样的范围，但只用一半显存。
> A100/H100 的 bf16 算力是 fp32 的 4-8x。""")

code("""# 演示混合精度 (CPU 上模拟)
# 注意: 真正的混合精度需要 GPU, 这里演示概念

torch.manual_seed(42)
model = SimpleModel()
x = torch.randn(32, 10)
y = torch.randn(32)

# fp32 计算
loss_fp32 = F.mse_loss(model(x), y)
print(f"fp32 损失: {loss_fp32.item():.6f}")

# fp16 计算 (模拟)
with torch.autocast(device_type='cpu', dtype=torch.float16):
    loss_fp16 = F.mse_loss(model(x), y)
print(f"fp16 损失: {loss_fp16.item():.6f}")
print(f"差异: {abs(loss_fp32.item() - loss_fp16.item()):.8f}")

# 显存对比
n_params = sum(p.numel() for p in model.parameters())
print(f"\\n模型参数: {n_params:,}")
print(f"fp32 显存: {n_params * 4 / 1024:.1f} KB")
print(f"fp16 显存: {n_params * 2 / 1024:.1f} KB (节省50%)")
print(f"bf16 显存: {n_params * 2 / 1024:.1f} KB (同fp16, 但范围更大)")""")

# ============================================================
md("""## 5. 完整训练循环：综合应用

把所有技巧组合起来——这是现代 LLM 训练的标准管线。""")

code("""def modern_train_loop(model, X, y, epochs=100, batch_size=32,
                       lr=3e-4, warmup_ratio=0.1, grad_clip=1.0,
                       accum_steps=1):
    \"\"\"现代训练循环: warmup+cosine + 梯度累积 + 梯度裁剪\"\"\"
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n_steps = epochs * (len(X) // batch_size)
    warmup_steps = int(n_steps * warmup_ratio)

    losses = []
    step = 0

    for epoch in range(epochs):
        idx = torch.randperm(len(X))
        optimizer.zero_grad()

        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i+batch_size]
            loss = F.mse_loss(model(X[batch_idx]), y[batch_idx]) / accum_steps
            loss.backward()

            if (i // batch_size + 1) % accum_steps == 0:
                # 学习率调度
                if step < warmup_steps:
                    lr_t = lr * step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (n_steps - warmup_steps)
                    lr_t = 0.5 * lr * (1 + np.cos(np.pi * progress))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_t

                # 梯度裁剪
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()
                step += 1
                losses.append(loss.item() * accum_steps)

    return losses

# 运行完整训练
torch.manual_seed(42)
model = SimpleModel()
X_train = torch.randn(256, 10)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).float()

losses = modern_train_loop(model, X_train, y_train, epochs=50, batch_size=32,
                           lr=3e-3, warmup_ratio=0.1, grad_clip=1.0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses, 'b-', linewidth=1.5, alpha=0.7)
# 滑动平均
window = 10
smoothed = [np.mean(losses[max(0,i-window):i+1]) for i in range(len(losses))]
ax.plot(smoothed, 'r-', linewidth=2.5, label='滑动平均')
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_title('完整训练循环: warmup+cosine+梯度裁剪')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_modern_training.png', bbox_inches='tight')
plt.show()
print(f"训练完成: 初始损失={losses[0]:.4f}, 最终损失={losses[-1]:.4f}")
print("现代训练循环: warmup稳定起步 → cosine平滑收尾 → 梯度裁剪防爆。")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| Warmup + Cosine Decay 学习率调度 | ✅ |
| 梯度累积模拟大 batch | ✅ |
| 梯度裁剪防爆炸 | ✅ |
| 混合精度训练 (fp16/bf16) | ✅ |
| 完整现代训练循环 | ✅ |

### 核心 takeaway

> **训练技巧是"免费的"性能提升**——不改模型，只改训练方式。
> Warmup+Cosine 是标准调度，梯度累积省显存，裁剪防爆，混合精度加速。
> 这些技巧组合起来就是现代 LLM 训练管线。

### 🔗 下一章预告

**`16_model_merging.ipynb`** — TIES/DARE/SLERP 模型合并

---

> 💬 **写在最后**：同样的模型，好的训练技巧能让效果提升 10-20%。
> 这些"细节"是区分好模型和差模型的关键。""")

# ============================================================
output_path = "notebooks/15_training_techniques.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")