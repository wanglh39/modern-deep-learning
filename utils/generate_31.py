# 生成 31_non_contrastive_ssl.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 31 — 非对比自监督：BYOL、DINO、DINOv2

> 对比学习需要负样本，但 BYOL 说**不需要**！
> DINO 用自蒸馏，DINOv2 达到监督学习水平。

## 本章你将掌握

1. **BYOL**：没有负样本的自监督
2. **DINO**：自蒸馏 + 学生-教师
3. **DINOv2**：大规模自监督的终极
4. **对比 vs 非对比**：为什么不需要负样本""")

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

md("""## 1. 对比学习的问题

### 1.1 负样本的麻烦

```
对比学习需要负样本:
  SimCLR: batch 内其他 → 需大 batch
  MoCo:   队列 → 需大队列
  CLIP:   batch 内其他配对 → 需大 batch

问题:
  1. 负样本数量影响效果 → 工程复杂
  2. 负样本质量 → 假负样本 (同类被当负样本)
  3. 大 batch/队列 → 显存/计算开销
```

### 1.2 BYOL 的惊人发现

BYOL（Bootstrap Your Own Latent）说：**不需要负样本！**

```
只需: 两个网络 (online + target)
  online: 有梯度, 正常训练
  target: 动量更新 (类似 MoCo)

损失: MSE(online(x1), target(x2))
  → 只让两个网络对同一图的不同增强输出一致
  → 不需要推开任何东西
```

> 💡 BYOL 的反直觉：没有负样本，模型为什么不会**坍缩**到常数解？
# 答案：BatchNorm 隐式提供了对比信号，加上动量更新的 target 网络。""")

md("""## 2. BYOL 架构

### 2.1 两个网络

```
Online 网络:
  encoder f_θ → projector g_θ → predictor q_θ

Target 网络:
  encoder f_ξ → projector g_ξ
  (ξ = 动量更新: τ·ξ + (1-τ)·θ)

流程:
  x1 → online → online_proj → q → pred1
  x2 → target → target_proj (stop gradient)
  loss = MSE(pred1, target_proj)

  对称: x2→online, x1→target, 取平均
```

### 2.2 Predictor

```
为什么需要 predictor q_θ?

没有 q:  online = target → 平凡解 (直接复制)
有 q:    online → q → 预测 target
         q 防止坍缩, 让 online 学有意义的表示
```

### 2.3 为什么不坍缩

```
1. 动量更新: target 慢慢变, online 追 → 不是直接复制
2. BatchNorm: 隐式标准化 → 防止输出全相同
3. Predictor: 非对称结构 → 防止平凡解
```

> 💡 BYOL 的关键洞察：**非对称结构**防止坍缩。
# online 有 predictor，target 没有——这种不对称让模型必须学有意义的表示。""")

code("""# BYOL 简化实现
class BYOL(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=256, momentum=0.99):
        super().__init__()
        self.momentum = momentum

        # Online 网络
        self.online_encoder = nn.Sequential(nn.Linear(784, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feat_dim))
        self.online_projector = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feat_dim))
        self.predictor = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feat_dim))

        # Target 网络 (动量更新)
        self.target_encoder = nn.Sequential(nn.Linear(784, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feat_dim))
        self.target_projector = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feat_dim))

        # 初始化 target = online
        for o, t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            t.data.copy_(o.data); t.requires_grad = False
        for o, t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            t.data.copy_(o.data); t.requires_grad = False

    def forward(self, x1, x2):
        # Online
        online_z1 = self.online_projector(self.online_encoder(x1))
        pred1 = self.predictor(online_z1)
        online_z2 = self.online_projector(self.online_encoder(x2))
        pred2 = self.predictor(online_z2)

        # Target (no grad)
        with torch.no_grad():
            target_z1 = self.target_projector(self.target_encoder(x1))
            target_z2 = self.target_projector(self.target_encoder(x2))

        # BYOL loss: MSE (对称)
        loss1 = 2 - 2 * F.cosine_similarity(pred1, target_z2.detach(), dim=-1).mean()
        loss2 = 2 - 2 * F.cosine_similarity(pred2, target_z1.detach(), dim=-1).mean()
        return (loss1 + loss2) / 2

    @torch.no_grad()
    def update_target(self):
        for o, t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            t.data = self.momentum * t.data + (1 - self.momentum) * o.data
        for o, t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            t.data = self.momentum * t.data + (1 - self.momentum) * o.data

# 训练 BYOL
def train_byol(n_epochs=50):
    model = BYOL(feat_dim=64, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.manual_seed(42)
    centers = torch.randn(10, 784)
    losses = []
    for _ in range(n_epochs):
        idx = torch.randint(0, 10, (64,))
        x1 = centers[idx] + 0.3 * torch.randn(64, 784)
        x2 = centers[idx] + 0.3 * torch.randn(64, 784)

        loss = model(x1, x2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target()
        losses.append(loss.item())

    return model, losses

model, losses = train_byol()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('BYOL Loss')
ax.set_title('BYOL 训练 (无负样本)'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_byol_train.png', bbox_inches='tight')
plt.show()
print(f"BYOL 训练: loss {losses[0]:.3f} → {losses[-1]:.3f}")
print("BYOL: 无负样本, 靠非对称结构防止坍缩。")""")

md("""## 3. DINO：自蒸馏

### 3.1 思想

DINO（self-DIstillation with NO labels）用**自蒸馏**：

```
学生网络: 有梯度, 正常训练
教师网络: 动量更新 (类似 BYOL)

输入: 同一图的不同裁剪
  全局裁剪 (大视野) → 教师
  局部裁剪 (小视野) → 学生

目标: 学生的局部裁剪预测教师的全局裁剪
  → 学生从局部推断全局 → 学到语义
```

### 3.2 关键创新

```
1. 中心化 + 锐化:
   教师输出中心化 (减均值) → 防止坍缩
   锐化 (温度低) → 更尖锐的分布

2. 多裁剪策略:
   2 个全局 + N 个局部裁剪
   学生看全部, 教师只看全局
   → 局部到全局的推理

3. 交叉熵损失 (不是 MSE):
   把输出当概率分布, 用 KL 散度
```

### 3.3 DINO 的涌现特性

DINO 学到的表示展现了惊人特性：
- **注意力图**自动定位物体
- **k-NN 分类**不需要微调就很好
- 表示空间的**线性可分性**

> 💡 DINO 的注意力图会自动高亮物体——无监督学到了物体定位。
# 这种"涌现"特性说明自监督学到了真正的语义。""")

code("""# DINO 简化实现
class DINO(nn.Module):
    def __init__(self, feat_dim=128, momentum=0.99, n_local=8):
        super().__init__()
        self.momentum = momentum
        self.n_local = n_local

        # 学生网络
        self.student = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, feat_dim))
        self.student_head = nn.Linear(feat_dim, 1000)  # 假设 1000 "类"

        # 教师网络
        self.teacher = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, feat_dim))
        self.teacher_head = nn.Linear(feat_dim, 1000)

        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data.copy_(s.data); t.requires_grad = False

        # 中心化缓冲
        self.register_buffer('center', torch.zeros(1000))

    def forward(self, global_views, local_views):
        # 教师: 只看全局
        with torch.no_grad():
            teacher_out = [self.teacher_head(self.teacher(v)) for v in global_views]
            teacher_out = [F.softmax((t - self.center) / 0.04, dim=-1) for t in teacher_out]  # 锐化

        # 学生: 看全局 + 局部
        student_global = [self.student_head(self.student(v)) for v in global_views]
        student_local = [self.student_head(self.student(v)) for v in local_views]

        # 交叉熵损失: 学生匹配教师
        loss = 0
        for s in student_local:  # 局部 → 全局
            for t in teacher_out:
                loss += F.cross_entropy(s / 0.1, t.detach())
        loss /= (len(student_local) * len(teacher_out))

        return loss

    @torch.no_grad()
    def update_teacher(self):
        for s, t in zip(self.student.parameters(), self.teacher.parameters()):
            t.data = self.momentum * t.data + (1 - self.momentum) * s.data

# 演示 DINO
dino = DINO(feat_dim=64, n_local=4)
global_views = [torch.randn(16, 784) for _ in range(2)]
local_views = [torch.randn(16, 784) for _ in range(4)]
loss = dino(global_views, local_views)
dino.update_teacher()

print(f"全局裁剪: {len(global_views)} 个 (教师+学生)")
print(f"局部裁剪: {len(local_views)} 个 (只学生)")
print(f"Loss: {loss.item():.3f}")
print("DINO: 学生从局部推断全局——学到语义表示。")""")

md("""## 4. DINOv2：大规模自监督

### 4.1 目标

DINOv2（2023）目标是：**自监督达到监督学习水平**。

### 4.2 关键改进

```
1. 数据: 1.42 亿张图 (LVD-142M)
   - 自动爬取 + 去重 + 质量筛选

2. 架构: ViT-g (10亿参数)
   - 比 DINO 大很多

3. 训练技巧:
   - iBOT 损失 (DINO + 掩码建模)
   - KoLeo 正则化 (均匀分布)
   - 高分辨率微调

4. 工程: Flash Attention, FSDP
   - 让 10 亿参数训练可行
```

### 4.3 结果

DINOv2 的表示：
- **线性探测**：超过监督学习
- **k-NN 分类**：接近监督学习
- **下游任务**：分割/检测不需要微调
- **跨域泛化**：深度估计、语义分割

> 💡 DINOv2 是自监督学习的里程碑——**第一次**在通用表示上超过监督学习。
# 以后可能不需要 ImageNet 监督预训练了。""")

code("""# DINOv2 vs 其他方法对比
methods = ['监督\\\\n(ImageNet)', 'SimCLR', 'BYOL', 'DINO', 'DINOv2']
linear_probe = [83.0, 75.3, 78.4, 80.1, 86.5]  # 线性探测准确率
knn = [78.0, 71.0, 74.0, 77.0, 83.5]  # k-NN 准确率

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, linear_probe, width, label='线性探测', color='steelblue', alpha=0.8)
ax.bar(x + width/2, knn, width, label='k-NN', color='coral', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('ImageNet 准确率 (%)')
ax.set_title('自监督方法对比 (DINOv2 超越监督)')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(60, 90)
plt.tight_layout()
plt.savefig('notebooks/fig_dinov2.png', bbox_inches='tight')
plt.show()
print("DINOv2: 自监督第一次超过监督学习——里程碑。")""")

md("""## 5. 对比 vs 非对比

### 5.1 为什么非对比也能工作

```
对比学习:
  显式推开负样本 → 防止坍缩
  需要: 负样本 (大 batch/队列)

非对比学习:
  隐式防止坍缩 → 靠架构设计
  BYOL: 非对称 (predictor) + 动量
  DINO: 中心化 + 锐化 + 多裁剪
  → 不需要负样本
```

### 5.2 掩码自编码器 (MAE)

另一条路线：**掩码建模**

```
MAE:
  遮盖 75% 的 patch → 只编码 25% → 重建遮盖部分

特点:
  - 不需要正负样本
  - 极高掩码率 (75%) → 强预训练任务
  - ViT 友好 (patch 天然适合)
```

### 5.3 统一视角

```
自监督学习的三大范式:

1. 对比学习 (SimCLR, MoCo): 拉正推负
2. 非对比 (BYOL, DINO): 自蒸馏, 防坍缩
3. 掩码建模 (MAE, BEiT): 重建遮盖

→ 都在学习"数据自身的结构"
```

> 💡 三大范式殊途同归：都是让模型理解数据自身的结构。
# DINOv2 证明：自监督可以超过监督——数据本身就是最好的标签。""")

code("""# 三大范式对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 对比学习
ax = axes[0]
ax.set_title('对比学习\\\\n(SimCLR, MoCo)', fontsize=12)
ax.text(0.5, 0.8, '拉近正样本', ha='center', fontsize=11, color='green')
ax.text(0.5, 0.6, '推开负样本', ha='center', fontsize=11, color='red')
ax.text(0.5, 0.3, '需要: 大batch/队列', ha='center', fontsize=10, color='gray')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# 非对比
ax = axes[1]
ax.set_title('非对比\\\\n(BYOL, DINO)', fontsize=12)
ax.text(0.5, 0.8, '自蒸馏', ha='center', fontsize=11, color='blue')
ax.text(0.5, 0.6, '防坍缩设计', ha='center', fontsize=11, color='purple')
ax.text(0.5, 0.3, '需要: 非对称结构', ha='center', fontsize=10, color='gray')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# 掩码建模
ax = axes[2]
ax.set_title('掩码建模\\\\n(MAE, BEiT)', fontsize=12)
ax.text(0.5, 0.8, '遮盖+重建', ha='center', fontsize=11, color='orange')
ax.text(0.5, 0.6, '预测被遮盖的', ha='center', fontsize=11, color='brown')
ax.text(0.5, 0.3, '需要: 高掩码率', ha='center', fontsize=10, color='gray')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

plt.tight_layout()
plt.savefig('notebooks/fig_ssl_paradigms.png', bbox_inches='tight')
plt.show()
print("三大范式殊途同归: 都在学习数据自身结构。")""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| BYOL 无负样本 | ✅ |
| 防坍缩机制 | ✅ |
| DINO 自蒸馏 | ✅ |
| DINOv2 超监督 | ✅ |
| 三大自监督范式 | ✅ |

### 核心 takeaway
> **BYOL 证明负样本不是必须的，DINOv2 证明自监督可以超过监督**。
# 自监督学习的核心：让模型理解数据自身的结构。
# 未来：监督预训练可能被自监督完全取代。

### 🔗 下一章
**`32_embedding_retrieval.ipynb`** — 多向量 embedding、向量检索

---

> 💬 **板块五进行中。**""")

output_path = "notebooks/31_non_contrastive_ssl.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")