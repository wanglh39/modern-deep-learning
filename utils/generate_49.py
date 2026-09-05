# 生成 49_adversarial_robustness.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 49 — 对抗鲁棒性：FGSM、PGD 与对抗训练

> 对抗样本：人眼无差异，模型却完全误判。这是深度学习的"阿喀琉斯之踵"。

## 本章你将掌握

1. **对抗样本**：什么是？为什么存在？
2. **FGSM**：快速梯度符号法
3. **PGD**：投影梯度下降法
4. **对抗训练**：用对抗样本训练鲁棒模型
5. **认证防御**：可证明的鲁棒性""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
torch.manual_seed(42)
print("环境准备完成 ✅")""")

md("""## 1. 对抗样本

### 1.1 什么是对抗样本？

```
对抗样本 (Adversarial Example):
  x' = x + δ
  其中 ||δ|| 很小 (人眼看不出差异)
  但 f(x') ≠ f(x) (模型预测改变)

经典案例:
  熊猫图片 + 不可见扰动 → 模型认为是长臂猿
  停止标志 + 贴纸 → 模型认为是限速标志
```

### 1.2 为什么存在对抗样本？

```
原因1: 线性特性
  高维空间中，即使每个维度扰动很小
  累积起来 w·δ 可以很大
  → 线性模型也有对抗样本!

原因2: 决策边界
  决策边界是高维曲面
  任何点都离边界很近 (高维空间特性)
  → 小扰动就能跨过边界

原因3: 特征不稳定
  模型学到的不稳定特征
  人眼不敏感但模型高度依赖
```

> 💡 对抗样本不是 bug，而是高维线性空间的内在特性。""")

md("""## 2. FGSM：快速梯度符号法

### 2.1 算法

```
FGSM (Fast Gradient Sign Method):
  x' = x + ε * sign(∇_x L(f(x), y))

直觉:
  - 梯度方向是损失增加最快的方向
  - sign 取符号 → 每个维度都往增加损失的方向走
  - ε 控制扰动大小

优点: 极快 (一次梯度计算)
缺点: 扰动不是最优 (只是贪心)
```

### 2.2 实现""")

code("""# 构建一个简单的分类器
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleClassifier()

# 生成一个简单的"图片" (模拟 MNIST 数字)
torch.manual_seed(42)
x = torch.randn(1, 28*28) * 0.5 + 0.5
x = x.clamp(0, 1).view(1, 1, 28, 28)
y = torch.tensor([3])  # 假设是数字3

# FGSM 攻击
def fgsm_attack(model, x, y, epsilon):
    x.requires_grad = True
    output = model(x)
    loss = F.cross_entropy(output, y)
    model.zero_grad()
    loss.backward()

    perturbation = epsilon * x.grad.sign()
    x_adv = x + perturbation
    return x_adv.detach(), perturbation.detach()

# 测试不同 epsilon
epsilons = [0, 0.05, 0.1, 0.2, 0.3]
print("FGSM 攻击效果:")
print("=" * 50)
for eps in epsilons:
    x_adv, pert = fgsm_attack(model, x, y, eps)
    with torch.no_grad():
        orig_pred = model(x).argmax().item()
        adv_pred = model(x_adv).argmax().item()
        orig_conf = F.softmax(model(x), dim=1)[0, y].item()
        adv_conf = F.softmax(model(x_adv), dim=1)[0, y].item()

    status = "✅ 正确" if adv_pred == y.item() else "❌ 误判"
    print(f"ε={eps:.2f}: 原始预测={orig_pred} (置信度{orig_conf:.3f}) → "
          f"对抗预测={adv_pred} (置信度{adv_conf:.3f}) {status}")""")

md("""## 3. PGD：投影梯度下降法

### 3.1 算法

```
PGD (Projected Gradient Descent):
  x_0 = x + random_perturbation  # 随机初始化
  for t in range(num_steps):
      gradient = ∇_x L(f(x_t), y)
      x_{t+1} = x_t + α * sign(gradient)  # 走一步
      x_{t+1} = clip(x_{t+1}, x-ε, x+ε)   # 投影到 ε-球
      x_{t+1} = clip(x_{t+1}, 0, 1)       # 投影到合法范围

vs FGSM:
  FGSM: 一步 → 快但不优
  PGD: 多步 → 慢但更强

PGD 是"最强一阶攻击" → 用 PGD 做对抗训练最可靠
```

### 3.2 实现""")

code("""# PGD 攻击
def pgd_attack(model, x, y, epsilon, alpha, num_steps):
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0, 1)

    for _ in range(num_steps):
        x_adv.requires_grad = True
        output = model(x_adv)
        loss = F.cross_entropy(output, y)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            x_adv = x_adv.clamp(0, 1)

    return x_adv.detach()

# 对比 FGSM 和 PGD
epsilon = 0.2
alpha = 0.01
steps_list = [5, 10, 20, 40]

print(f"PGD 攻击 (ε={epsilon}, α={alpha}):")
print("=" * 50)

# FGSM 基准
x_fgsm, _ = fgsm_attack(model, x, y, epsilon)
with torch.no_grad():
    fgsm_pred = model(x_fgsm).argmax().item()
    fgsm_loss = F.cross_entropy(model(x_fgsm), y).item()
print(f"FGSM: 预测={fgsm_pred}, 损失={fgsm_loss:.4f}")

# PGD 不同步数
for steps in steps_list:
    x_pgd = pgd_attack(model, x, y, epsilon, alpha, steps)
    with torch.no_grad():
        pgd_pred = model(x_pgd).argmax().item()
        pgd_loss = F.cross_entropy(model(x_pgd), y).item()
    print(f"PGD ({steps:2d}步): 预测={pgd_pred}, 损失={pgd_loss:.4f}")

print("\\nPGD 步数越多，攻击越强 (损失越高)。")""")

md("""## 4. 对抗训练

### 4.1 核心思想

```
对抗训练 (Adversarial Training):
  min_θ E_{(x,y)} [ max_{||δ||≤ε} L(f_θ(x+δ), y) ]

直觉:
  外层: 最小化训练损失 (训练模型)
  内层: 最大化损失 (找最坏对抗样本)

  → 在最坏情况下训练 → 模型对对抗样本鲁棒

实现:
  for each batch:
      1. 生成对抗样本 (PGD)
      2. 用对抗样本训练模型
```

### 4.2 实现""")

code("""# 对抗训练
torch.manual_seed(42)

# 生成合成数据
n_samples = 500
X_data = torch.randn(n_samples, 1, 28, 28).clamp(0, 1)
y_data = torch.randint(0, 10, (n_samples,))
dataset = TensorDataset(X_data, y_data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 两个模型: 标准训练 vs 对抗训练
model_standard = SimpleClassifier()
model_robust = SimpleClassifier()

opt_std = optim.Adam(model_standard.parameters(), lr=1e-3)
opt_adv = optim.Adam(model_robust.parameters(), lr=1e-3)

epsilon = 0.1
alpha = 0.01
pgd_steps = 5

std_losses = []
adv_losses = []

for epoch in range(10):
    for batch_x, batch_y in loader:
        # 标准训练
        opt_std.zero_grad()
        loss_std = F.cross_entropy(model_standard(batch_x), batch_y)
        loss_std.backward()
        opt_std.step()
        std_losses.append(loss_std.item())

        # 对抗训练
        model_robust.eval()
        x_adv = pgd_attack(model_robust, batch_x, batch_y, epsilon, alpha, pgd_steps)
        model_robust.train()

        opt_adv.zero_grad()
        loss_adv = F.cross_entropy(model_robust(x_adv), batch_y)
        loss_adv.backward()
        opt_adv.step()
        adv_losses.append(loss_adv.item())

# 评估鲁棒性
def evaluate_robustness(model, x, y, epsilons):
    results = []
    for eps in epsilons:
        x_adv = pgd_attack(model, x, y, eps, 0.01, 10)
        with torch.no_grad():
            clean_acc = (model(x).argmax(1) == y).float().mean().item()
            adv_acc = (model(x_adv).argmax(1) == y).float().mean().item()
        results.append((eps, clean_acc, adv_acc))
    return results

test_x = X_data[:50]
test_y = y_data[:50]
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.3]

std_results = evaluate_robustness(model_standard, test_x, test_y, epsilons)
adv_results = evaluate_robustness(model_robust, test_x, test_y, epsilons)

print("鲁棒性对比:")
print(f"{'ε':>6} | {'标准模型 干净/对抗':>20} | {'对抗训练 干净/对抗':>20}")
print("-" * 55)
for (eps, sc, sa), (_, ac, aa) in zip(std_results, adv_results):
    print(f"{eps:6.2f} | {sc:8.2f} / {sa:8.2f}     | {ac:8.2f} / {aa:8.2f}")""")

code("""# 可视化对抗训练效果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 训练损失
ax = axes[0]
window = 20
std_smooth = np.convolve(std_losses, np.ones(window)/window, mode='valid')
adv_smooth = np.convolve(adv_losses, np.ones(window)/window, mode='valid')
ax.plot(std_smooth, 'b-', alpha=0.7, label='标准训练')
ax.plot(adv_smooth, 'r-', alpha=0.7, label='对抗训练')
ax.set_xlabel('训练步数', fontsize=12)
ax.set_ylabel('损失', fontsize=12)
ax.set_title('训练损失', fontsize=13, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)

# 鲁棒性曲线
ax = axes[1]
eps = [r[0] for r in std_results]
std_acc = [r[2] for r in std_results]
adv_acc = [r[2] for r in adv_results]
ax.plot(eps, std_acc, 'b-o', linewidth=2, markersize=8, label='标准模型')
ax.plot(eps, adv_acc, 'r-s', linewidth=2, markersize=8, label='对抗训练模型')
ax.set_xlabel('扰动大小 ε', fontsize=12)
ax.set_ylabel('对抗准确率', fontsize=12)
ax.set_title('对抗鲁棒性', fontsize=13, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_adversarial_training.png', bbox_inches='tight')
plt.show()
print("对抗训练: 在干净数据上略差，但在对抗扰动下显著更鲁棒。")""")

md("""## 5. 认证防御

### 5.1 经验防御 vs 认证防御

```
经验防御 (Empirical):
  - 用已知攻击测试
  - 对未知攻击可能失效
  - FGSM/PGD 对抗训练属于此类

认证防御 (Certified):
  - 可证明在 ε-球内不会误判
  - 对任何攻击都有效
  - 但通常更保守 (准确率更低)
```

### 5.2 随机平滑 (Randomized Smoothing)

```
随机平滑:
  f_smooth(x) = argmax_c E[f(x + δ)]  where δ ~ N(0, σ²I)

  → f_smooth 是 L2-Lipschitz 连续的
  → 可证明 ||x - x'|| ≤ R 时 f_smooth(x) = f_smooth(x')

优点: 适用于任意模型
缺点: 平滑后准确率下降
```

> 💡 认证防御给出可证明的保证，但通常以准确率为代价。""")

code("""# 随机平滑简化实现
def randomized_smoothing_predict(model, x, sigma, num_samples, num_classes):
    # 随机平滑预测
    counts = np.zeros(num_classes)
    for _ in range(num_samples):
        noise = torch.randn_like(x) * sigma
        with torch.no_grad():
            pred = model(x + noise).argmax().item()
        counts[pred] += 1
    return counts.argmax(), counts

# 测试不同 sigma
sigmas = [0, 0.1, 0.25, 0.5, 1.0]
print("随机平滑效果:")
print("=" * 50)
for sigma in sigmas:
    pred, counts = randomized_smoothing_predict(model_standard, x, sigma, 100, 10)
    confidence = counts.max() / counts.sum()
    print(f"σ={sigma:.2f}: 预测={pred}, 置信度={confidence:.2%}")

# 认证半径 (简化)
print("\\n认证半径 R ≈ σ * Φ⁻¹(p_correct)")
print("σ 越大 → R 越大 (更鲁棒) 但准确率越低")""")

md("""## 6. 对抗鲁棒性的权衡

### 6.1 鲁棒性 vs 准确率

```
权衡 (Trade-off):
  ε=0:   最高准确率，零鲁棒性
  ε=∞:   最高鲁棒性，随机准确率

  → 需要选择合适的 ε

  经验:
  - ε=8/255 (L∞): 常用设定
  - ε=2/255: 轻微扰动
  - ε=32/255: 大扰动 (人眼可见)
```

### 6.2 鲁棒性 vs 泛化

```
有趣发现:
  鲁棒模型 → 更好的泛化
  鲁棒特征 → 更符合人类直觉

  → 对抗训练不仅防攻击，还可能提升模型质量
```

> 💡 对抗鲁棒性不是纯防御——鲁棒特征往往更可解释、更符合人类直觉。""")

code("""# 鲁棒性 vs 准确率权衡
fig, ax = plt.subplots(figsize=(10, 6))

epsilons = np.linspace(0, 0.5, 50)
clean_acc = 0.95 * np.exp(-epsilons * 0.5)
robust_acc = 0.85 * np.exp(-epsilons * 1.5) + 0.1

ax.fill_between(epsilons, robust_acc, clean_acc, alpha=0.2, color='gray',
                label='鲁棒性间隙')
ax.plot(epsilons, clean_acc, 'b-', linewidth=2, label='干净准确率 (标准模型)')
ax.plot(epsilons, robust_acc, 'r-', linewidth=2, label='对抗准确率 (对抗训练)')

ax.set_xlabel('扰动大小 ε', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('鲁棒性 vs 准确率权衡', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
ax.set_ylim(0, 1.0)

# 标注常用设定
ax.axvline(x=8/255, color='green', linestyle='--', alpha=0.5)
ax.annotate('常用设定 ε=8/255', xy=(8/255, 0.5), fontsize=10, color='green',
            ha='left', rotation=90)

plt.tight_layout()
plt.savefig('notebooks/fig_robustness_tradeoff.png', bbox_inches='tight')
plt.show()
print("ε 越大，对抗训练越鲁棒，但干净准确率下降 → 需要权衡。")""")

md("""## 7. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 对抗样本存在性 | ✅ |
| FGSM 快速攻击 | ✅ |
| PGD 最强一阶攻击 | ✅ |
| 对抗训练 | ✅ |
| 认证防御 (随机平滑) | ✅ |
| 鲁棒性 vs 准确率权衡 | ✅ |

### 核心 takeaway
> **对抗样本是高维空间的内在特性**——FGSM/PGD 发现攻击，对抗训练提升鲁棒性，认证防御提供保证。鲁棒性与准确率的权衡是核心挑战。

### 🔗 下一章
**`50_ood_calibration.ipynb`** — OOD 检测、温度缩放、置信度校准

---

> 💬 **板块八(对齐、安全与评估)进行中 (3/5)。**""")

output_path = "notebooks/49_adversarial_robustness.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")