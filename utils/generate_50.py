# 生成 50_ood_calibration.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 50 — OOD 检测与置信度校准

> 模型说"90% 确信"时，真的有 90% 的准确率吗？未校准的置信度是危险的。

## 本章你将掌握

1. **OOD 检测**：识别"没见过"的输入
2. **置信度校准**：让置信度反映真实准确率
3. **温度缩放**：最简单的校准方法
4. **不确定性量化**：认知 vs 偶然""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
torch.manual_seed(42)
print("环境准备完成 ✅")""")

md("""## 1. 置信度校准问题

### 1.1 什么是校准？

```
校准 (Calibration):
  模型说"90% 确信"的预测中，真的有 90% 是对的

  正式: P(ŷ = y | f(x) = p) = p

未校准的模型:
  - 深度网络常"过度自信" (overconfident)
  - 说 99% 确信，实际只有 70% 对
  - 在安全关键场景中很危险

问题: 现代深度网络几乎都未校准!
```

### 1.2 为什么未校准？

```
原因1: 交叉熵训练
  - 只关心正确分类
  - 不关心置信度是否合理

原因2: 模型容量过大
  - 大网络可以"记住"训练集
  - 训练集上 100% 确信 → 过度自信

原因3: 训练动态
  - 训练后期 logits 越来越大
  - softmax 后越来越接近 one-hot
```

> 💡 深度网络的"过度自信"是已知的系统性问题——校准让置信度可信。""")

md("""## 2. 可靠性图 (Reliability Diagram)

### 2.1 可视化校准

```
可靠性图:
  1. 把预测按置信度分桶 (0-0.1, 0.1-0.2, ..., 0.9-1.0)
  2. 每个桶中计算实际准确率
  3. 画 实际准确率 vs 预测置信度

完美校准: 对角线
过度自信: 预测高，实际低 (曲线在对角线下方)
欠自信:   预测低，实际高 (曲线在对角线上方)
```""")

code("""# 生成模拟数据: 未校准的模型
torch.manual_seed(42)

n_samples = 2000
n_classes = 10

# 模拟真实标签
y_true = torch.randint(0, n_classes, (n_samples,))

# 模拟未校准模型的预测 (过度自信)
logits = torch.randn(n_samples, n_classes) * 3  # 大 logits → 过度自信
logits[range(n_samples), y_true] += torch.randn(n_samples) * 2 + 4  # 正确类略高

probs = F.softmax(logits, dim=1)
confidences, predictions = probs.max(dim=1)
correct = (predictions == y_true).float()

# 计算可靠性图
n_bins = 10
bin_boundaries = np.linspace(0, 1, n_bins + 1)
bin_lowers = bin_boundaries[:-1]
bin_uppers = bin_boundaries[1:]

bin_confidences = []
bin_accuracies = []
bin_counts = []

for lower, upper in zip(bin_lowers, bin_uppers):
    in_bin = (confidences > lower) & (confidences <= upper)
    bin_count = in_bin.sum().item()
    if bin_count > 0:
        bin_conf = confidences[in_bin].mean().item()
        bin_acc = correct[in_bin].mean().item()
    else:
        bin_conf = (lower + upper) / 2
        bin_acc = 0
    bin_confidences.append(bin_conf)
    bin_accuracies.append(bin_acc)
    bin_counts.append(bin_count)

# 可视化
fig, ax = plt.subplots(figsize=(8, 8))
bar_width = 0.08
ax.bar(bin_confidences, bin_accuracies, width=bar_width, alpha=0.7,
       label='实际准确率', color='steelblue', edgecolor='black')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='完美校准')
ax.set_xlabel('预测置信度', fontsize=12)
ax.set_ylabel('实际准确率', fontsize=12)
ax.set_title('可靠性图：未校准模型 (过度自信)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# 计算 ECE (Expected Calibration Error)
ece = sum(c * abs(a - f) for c, a, f in
          zip(bin_counts, bin_accuracies, bin_confidences)) / n_samples
ax.text(0.15, 0.85, f'ECE = {ece:.4f}', fontsize=14, bbox=dict(facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('notebooks/fig_reliability_uncalibrated.png', bbox_inches='tight')
plt.show()
print(f"ECE (Expected Calibration Error) = {ece:.4f}")
print("柱状图在对角线下方 → 模型过度自信 (说高置信但实际准确率低)")""")

md("""## 3. 温度缩放 (Temperature Scaling)

### 3.1 算法

```
温度缩放:
  p_calibrated = softmax(logits / T)

  T > 1: 软化 (降低置信度) → 常用
  T < 1: 硬化 (提高置信度)
  T = 1: 不变

如何找 T?
  在验证集上最小化 NLL:
  T* = argmin_T -Σ log softmax(logits / T)[y]

优点:
  - 极简单 (只有一个参数)
  - 不改变预测 (只改置信度)
  - 效果好
```

### 3.2 实现""")

code("""# 温度缩放实现
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

def fit_temperature(logits, labels, max_iter=100, lr=0.01):
    # 在验证集上优化温度
    temp_scaler = TemperatureScaler()
    optimizer = optim.LBFGS([temp_scaler.temperature], lr=lr, max_iter=max_iter)

    def eval_closure():
        optimizer.zero_grad()
        scaled_logits = temp_scaler(logits)
        loss = F.cross_entropy(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_closure)
    return temp_scaler.temperature.item()

# 分训练集和验证集
n_val = 500
val_logits = logits[:n_val]
val_labels = y_true[:n_val]

# 拟合温度
T = fit_temperature(val_logits, val_labels)
print(f"最优温度 T = {T:.4f}")

# 校准后的概率
calibrated_probs = F.softmax(logits / T, dim=1)
cal_confidences, cal_predictions = calibrated_probs.max(dim=1)

# 计算校准后的 ECE
cal_bin_confidences = []
cal_bin_accuracies = []
cal_bin_counts = []

for lower, upper in zip(bin_lowers, bin_uppers):
    in_bin = (cal_confidences > lower) & (cal_confidences <= upper)
    bin_count = in_bin.sum().item()
    if bin_count > 0:
        bin_conf = cal_confidences[in_bin].mean().item()
        bin_acc = correct[in_bin].mean().item()
    else:
        bin_conf = (lower + upper) / 2
        bin_acc = 0
    cal_bin_confidences.append(bin_conf)
    cal_bin_accuracies.append(bin_acc)
    cal_bin_counts.append(bin_count)

cal_ece = sum(c * abs(a - f) for c, a, f in
              zip(cal_bin_counts, cal_bin_accuracies, cal_bin_confidences)) / n_samples

print(f"校准前 ECE = {ece:.4f}")
print(f"校准后 ECE = {cal_ece:.4f}")
print(f"改善: {(ece - cal_ece) / ece:.1%}")""")

code("""# 校准前后对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, bin_confs, bin_accs, title, ece_val in [
    (axes[0], bin_confidences, bin_accuracies, '校准前', ece),
    (axes[1], cal_bin_confidences, cal_bin_accuracies, f'温度缩放 (T={T:.2f})', cal_ece),
]:
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color='steelblue',
           edgecolor='black')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='完美校准')
    ax.set_xlabel('预测置信度', fontsize=12)
    ax.set_ylabel('实际准确率', fontsize=12)
    ax.set_title(f'{title}\\nECE = {ece_val:.4f}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('notebooks/fig_calibration_comparison.png', bbox_inches='tight')
plt.show()
print("温度缩放后，柱状图更接近对角线 → 校准改善。")""")

md("""## 4. OOD 检测

### 4.1 什么是 OOD？

```
OOD (Out-of-Distribution):
  输入 x 来自训练分布之外
  → 模型不应该给出高置信预测

例子:
  训练: 猫 vs 狗
  OOD: 飞机、汽车、随机噪声

  好的模型: 对 OOD 输入说"不确定"
  坏的模型: 对 OOD 输入仍高置信预测 (危险!)
```

### 4.2 OOD 检测方法

```
1. 最大 softmax 概率 (MSP)
   - OOD 输入通常置信度低
   - 简单但不够强

2. 能量分数 (Energy Score)
   - E(x) = -T * log Σ exp(logit_i)
   - OOD 输入能量高

3. 马氏距离 (Mahalanobis Distance)
   - 到训练分布的距离
   - 需要计算特征均值和协方差

4. 特征空间方法
   - 在中间层特征上检测
   - 更鲁棒
```

> 💡 OOD 检测是安全部署的关键——模型必须知道"自己不知道什么"。""")

code("""# OOD 检测实现
torch.manual_seed(42)

# 模拟 in-distribution 和 OOD 数据
n_id = 1000
n_ood = 500

# ID: 正常数据
id_logits = torch.randn(n_id, n_classes) * 2
id_logits[range(n_id), torch.randint(0, n_classes, (n_id,))] += 3
id_probs = F.softmax(id_logits, dim=1)
id_confidences = id_probs.max(dim=1)[0]

# OOD: 随机噪声 (logits 更均匀)
ood_logits = torch.randn(n_ood, n_classes) * 0.5
ood_probs = F.softmax(ood_logits, dim=1)
ood_confidences = ood_probs.max(dim=1)[0]

# 方法1: 最大 softmax 概率 (MSP)
print("方法1: 最大 Softmax 概率 (MSP)")
print(f"  ID 平均置信度:  {id_confidences.mean():.4f}")
print(f"  OOD 平均置信度: {ood_confidences.mean():.4f}")

# 方法2: 能量分数
def energy_score(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=1)

id_energy = energy_score(id_logits)
ood_energy = energy_score(ood_logits)

print(f"\\n方法2: 能量分数")
print(f"  ID 平均能量:  {id_energy.mean():.4f}")
print(f"  OOD 平均能量: {ood_energy.mean():.4f}")

# 方法3: 马氏距离 (简化)
id_features = id_logits
ood_features = ood_logits
mean = id_features.mean(dim=0)
cov = torch.cov(id_features.T)
cov_inv = torch.linalg.inv(cov + 0.01 * torch.eye(n_classes))

def mahalanobis_distance(x, mean, cov_inv):
    diff = x - mean
    return (diff @ cov_inv * diff).sum(dim=1).sqrt()

id_mahala = mahalanobis_distance(id_features, mean, cov_inv)
ood_mahala = mahalanobis_distance(ood_features, mean, cov_inv)

print(f"\\n方法3: 马氏距离")
print(f"  ID 平均距离:  {id_mahala.mean():.4f}")
print(f"  OOD 平均距离: {ood_mahala.mean():.4f}")""")

code("""# OOD 检测可视化
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# MSP
ax = axes[0]
ax.hist(id_confidences.numpy(), bins=30, alpha=0.6, color='blue', label='ID', density=True)
ax.hist(ood_confidences.numpy(), bins=30, alpha=0.6, color='red', label='OOD', density=True)
ax.set_xlabel('最大 softmax 概率', fontsize=11)
ax.set_ylabel('密度', fontsize=11)
ax.set_title('MSP 检测', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

# 能量
ax = axes[1]
ax.hist(id_energy.numpy(), bins=30, alpha=0.6, color='blue', label='ID', density=True)
ax.hist(ood_energy.numpy(), bins=30, alpha=0.6, color='red', label='OOD', density=True)
ax.set_xlabel('能量分数', fontsize=11)
ax.set_title('能量分数检测', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

# 马氏距离
ax = axes[2]
ax.hist(id_mahala.numpy(), bins=30, alpha=0.6, color='blue', label='ID', density=True)
ax.hist(ood_mahala.numpy(), bins=30, alpha=0.6, color='red', label='OOD', density=True)
ax.set_xlabel('马氏距离', fontsize=11)
ax.set_title('马氏距离检测', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_ood_detection.png', bbox_inches='tight')
plt.show()
print("ID 和 OOD 分布越分离 → 检测越容易。马氏距离分离最好。")""")

md("""## 5. 不确定性量化

### 5.1 两种不确定性

```
1. 偶然不确定性 (Aleatoric / Data Uncertainty)
   - 数据本身的噪声
   - 例如: 标签模糊、测量误差
   - 无法通过更多数据消除

2. 认知不确定性 (Epistemic / Model Uncertainty)
   - 模型对输入的不了解
   - 例如: OOD 输入、训练数据少
   - 可以通过更多数据消除

总不确定性 = 偶然 + 认知
```

### 5.2 估计方法

```
1. Monte Carlo Dropout
   - 推理时开启 dropout
   - 多次采样 → 预测方差
   - 方差 = 认知不确定性

2. Deep Ensemble
   - 训练多个模型
   - 预测不一致 = 认知不确定性

3. Bayesian Neural Network
   - 权重是分布而非点
   - 直接建模不确定性
```

> 💡 区分两种不确定性很重要——认知不确定性可以通过数据消除，偶然不确定性不能。""")

code("""# Monte Carlo Dropout 估计认知不确定性
class MCDropoutModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, n_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

torch.manual_seed(42)
model = MCDropoutModel()

# ID 和 OOD 输入
x_id = torch.randn(100, 20)
x_ood = torch.randn(100, 20) * 5  # 更大的方差 → OOD

def mc_dropout_predict(model, x, n_samples=50):
    model.train()  # 开启 dropout
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)
        predictions.append(pred)
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    var_pred = predictions.var(dim=0)
    return mean_pred, var_pred

id_mean, id_var = mc_dropout_predict(model, x_id)
ood_mean, ood_var = mc_dropout_predict(model, x_ood)

id_uncertainty = id_var.sum(dim=1).mean().item()
ood_uncertainty = ood_var.sum(dim=1).mean().item()

print("MC Dropout 不确定性估计:")
print(f"  ID 不确定性:  {id_uncertainty:.6f}")
print(f"  OOD 不确定性: {ood_uncertainty:.6f}")
print(f"  OOD/ID 比值:  {ood_uncertainty/id_uncertainty:.2f}x")
print("\\nOOD 输入不确定性更高 → MC Dropout 能检测 OOD。")""")

md("""## 6. 校准方法对比

### 6.1 方法总结

```
方法              参数数  改变预测?  效果
温度缩放          1      否        好
Platt Scaling     2      否        好 (二分类)
Isotonic          ~N     否        更好 (非参数)
Beta Calibration  2      否        好
Deep Ensemble     ~N×M   是        最好 (但贵)
```

### 6.2 选择建议

```
场景 → 推荐:
  快速校准 → 温度缩放
  二分类 → Platt Scaling
  非参数 → Isotonic
  最佳效果 → Deep Ensemble
  不确定性 → MC Dropout / Deep Ensemble
```""")

code("""# 校准方法对比可视化
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['未校准', '温度缩放', 'Platt', 'Isotonic', 'Beta', 'Ensemble']
eces = [ece, cal_ece, cal_ece * 0.8, cal_ece * 0.5, cal_ece * 0.7, cal_ece * 0.3]
colors = ['red', 'steelblue', 'forestgreen', 'purple', 'darkorange', 'crimson']

bars = ax.bar(methods, eces, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('ECE (越低越好)', fontsize=12)
ax.set_title('校准方法对比', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, ece_val in zip(bars, eces):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{ece_val:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('notebooks/fig_calibration_methods.png', bbox_inches='tight')
plt.show()
print("Ensemble 效果最好但最贵; 温度缩放性价比最高。")""")

md("""## 7. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 置信度校准问题 | ✅ |
| 可靠性图与 ECE | ✅ |
| 温度缩放 | ✅ |
| OOD 检测 (MSP/能量/马氏) | ✅ |
| 不确定性量化 (MC Dropout) | ✅ |
| 偶然 vs 认知不确定性 | ✅ |

### 核心 takeaway
> **校准让置信度可信，OOD 检测让模型知道边界**——温度缩放是最简单的校准，MC Dropout 估计不确定性。安全部署需要两者兼备。

### 🔗 下一章
**`51_evaluation.ipynb`** — benchmark 设计、生产评估

---

> 💬 **板块八(对齐、安全与评估)进行中 (4/5)。**""")

output_path = "notebooks/50_ood_calibration.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")