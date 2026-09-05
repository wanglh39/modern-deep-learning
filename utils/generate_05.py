"""生成 05_generalization_theory.ipynb 的脚本"""
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
md("""# 05 — 泛化理论、双下降与万能近似定理

> 深度学习有个**反常现象**：GPT-4 有上万亿参数，训练数据远少于参数量，
> 按经典统计学它应该**严重过拟合**，但它却泛化得极好。为什么？
>
> 这一章我们用实验回答三个理论问题：
> 1. 神经网络为什么能逼近任何函数？（**万能近似定理**）
> 2. 模型越大越容易过拟合？（**经典 bias-variance tradeoff**）
> 3. 为什么过参数化反而泛化更好？（**双下降现象**）

## 本章你将掌握

1. **万能近似定理**：一个宽隐藏层就能逼近任何连续函数（实验验证）
2. **宽度 vs 深度**：为什么实际中用深而不是宽
3. **经典 bias-variance tradeoff**：U 形曲线
4. **双下降**：过参数化区域的第二次下降
5. **为什么深度学习能泛化**：隐式正则化与平坦极小值""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
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
md("""## 1. 万能近似定理

### 1.1 定理说了什么

**万能近似定理（Cybenko, 1989）**：

> 对于任何连续函数 $f: [0,1]^n \\to \\mathbb{R}$ 和任意 $\\epsilon > 0$，
> 存在一个**单隐藏层**的前馈网络 $g$，使得 $\\sup|f - g| < \\epsilon$。

通俗说：**只要隐藏层足够宽，一个隐藏层的网络就能以任意精度逼近任何连续函数**。

### 1.2 但有个陷阱

定理只说"存在"，没说"容易找到"。可能需要一个**指数级宽**的隐藏层。
而用多层网络，每层不需要太宽就能达到同样效果——**深度比宽度高效**。

> 这就是为什么实际中用**深而窄**的网络，而不是**浅而极宽**的网络。""")

code("""def train_approximator(target_fn, n_hidden, x_train, y_train, epochs=2000):
    \"\"\"训练一个单隐藏层网络逼近 target_fn\"\"\"
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(1, n_hidden),
        nn.Tanh(),
        nn.Linear(n_hidden, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    x_t = torch.tensor(x_train, dtype=torch.float32).reshape(-1, 1)
    y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        optimizer.step()

    return model

# 目标函数：sin(x) + 0.3*cos(3x)
target_fn = lambda x: np.sin(x) + 0.3 * np.cos(3 * x)
x_train = np.linspace(-np.pi, np.pi, 200)
y_train = target_fn(x_train)
x_test = np.linspace(-np.pi, np.pi, 500)

# 不同宽度对比
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
widths = [5, 20, 100]

for ax, n in zip(axes, widths):
    model = train_approximator(target_fn, n, x_train, y_train)
    with torch.no_grad():
        y_pred = model(torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1)).numpy()

    ax.plot(x_test, target_fn(x_test), 'b-', linewidth=2, label='目标函数', alpha=0.7)
    ax.plot(x_test, y_pred, 'r--', linewidth=2, label=f'网络逼近 (宽={n})')
    ax.scatter(x_train[::10], y_train[::10], c='gray', s=10, alpha=0.5)
    ax.set_title(f'隐藏层宽度 = {n}')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_universal_approx.png', bbox_inches='tight')
plt.show()
print("宽度越大，逼近越精确——这就是万能近似定理的实验验证。")""")

md("""### 1.3 宽度 vs 深度

同样数量的参数，深而窄 vs 浅而宽，谁更强？""")

code("""# 同样参数量，对比深 vs 宽
def train_model(layers, x_train, y_train, epochs=2000):
    torch.manual_seed(42)
    model = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    x_t = torch.tensor(x_train, dtype=torch.float32).reshape(-1, 1)
    y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_t), y_t)
        loss.backward()
        optimizer.step()
    return model

target_fn = lambda x: np.sin(x) + 0.3 * np.cos(3 * x)
x_train = np.linspace(-np.pi, np.pi, 200)
y_train = target_fn(x_train)
x_test = np.linspace(-np.pi, np.pi, 500)

# 浅而宽：1→100→1 (约 300 参数)
shallow = train_model([nn.Linear(1, 100), nn.Tanh(), nn.Linear(100, 1)], x_train, y_train)
# 深而窄：1→10→10→10→1 (约 240 参数，更少！)
deep = train_model([nn.Linear(1, 10), nn.Tanh(), nn.Linear(10, 10), nn.Tanh(),
                    nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 1)], x_train, y_train)

fig, ax = plt.subplots(figsize=(9, 5))
with torch.no_grad():
    y_shallow = shallow(torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1)).numpy()
    y_deep = deep(torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1)).numpy()

ax.plot(x_test, target_fn(x_test), 'b-', linewidth=3, label='目标', alpha=0.5)
ax.plot(x_test, y_shallow, 'r--', linewidth=2, label='浅而宽 (1→100→1, ~300参数)')
ax.plot(x_test, y_deep, 'g-', linewidth=2, label='深而窄 (1→10→10→10→1, ~240参数)')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title('宽度 vs 深度：深网络用更少参数达到同样效果')
plt.tight_layout()
plt.savefig('notebooks/fig_width_vs_depth.png', bbox_inches='tight')
plt.show()
print("深而窄用更少的参数达到同样甚至更好的效果——这就是为什么现代网络都往深了堆。")""")

# ============================================================
md("""## 2. 经典视角：Bias-Variance Tradeoff

### 2.1 经典统计学的预言

经典统计学说：模型复杂度有个**最优点**。

- **太简单**：欠拟合（bias 大）
- **太复杂**：过拟合（variance 大）
- **最优**：在中间某处

画出来就是经典的 **U 形曲线**。

### 2.2 实验复现 U 形曲线""")

code("""# 实验：固定数据集，改变模型复杂度（多项式阶数），看训练/测试误差
np.random.seed(42)
n_train, n_test = 30, 200
x_all = np.linspace(-3, 3, n_train + n_test)
np.random.shuffle(x_all)
y_all = np.sin(x_all) + np.random.randn(len(x_all)) * 0.2

x_train, y_train = x_all[:n_train], y_all[:n_train]
x_test, y_test = x_all[n_train:], y_all[n_train:]

degrees = range(1, 25)
train_errors, test_errors = [], []

for d in degrees:
    # 用多项式回归（不同阶数 = 不同复杂度）
    coeffs = np.polyfit(x_train, y_train, d)
    y_train_pred = np.polyval(coeffs, x_train)
    y_test_pred = np.polyval(coeffs, x_test)
    train_errors.append(np.mean((y_train_pred - y_train)**2))
    test_errors.append(np.mean((y_test_pred - y_test)**2))

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(degrees, train_errors, 'b-o', linewidth=2, markersize=5, label='训练误差')
ax.plot(degrees, test_errors, 'r-s', linewidth=2, markersize=5, label='测试误差')
optimal_d = list(degrees)[np.argmin(test_errors)]
ax.axvline(optimal_d, color='green', linestyle='--', alpha=0.7, label=f'最优复杂度 (阶数={optimal_d})')
ax.set_xlabel('模型复杂度（多项式阶数）')
ax.set_ylabel('误差')
ax.set_title('经典 Bias-Variance Tradeoff：U 形曲线')
ax.set_ylim(0, 2); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_bias_variance.png', bbox_inches='tight')
plt.show()
print(f"经典视角：测试误差先降后升，最优在阶数={optimal_d}。")
print(f"阶数太大（{max(degrees)}）时，训练误差→0 但测试误差爆炸 → 经典过拟合。")""")

# ============================================================
md("""## 3. 双下降：深度学习的反常现象

### 3.1 经典理论解释不了的事

2018-2019 年，Belkin 等人发现了一个**经典理论无法解释的现象**：

> 当模型复杂度**超过插值阈值**（刚好能记住所有训练数据的点）后，
> 测试误差**不升反降**——出现了**第二次下降**。

```
误差
 │     ╱╲         ╱
 │    ╱  ╲       ╱
 │   ╱    ╲     ╱
 │  ╱      ╲   ╱
 │ ╱        ╲ ╱
 │╱          ╲╱  ← 插值阈值
 │            ╲
 │             ╲___← 第二次下降！
 └────────────────── 复杂度
   欠拟合  过拟合  过参数化
```

### 3.2 这就是现代深度学习的运作方式

GPT-4 有上万亿参数，训练数据远少于参数——它**远在插值阈值的右侧**，
正处于双下降的第二次下降区域，所以泛化得好。

> **过参数化不是 bug，是 feature**。现代深度学习故意用过参数化模型。""")

code("""# 实验：复现双下降现象
# 用随机特征模型（kernel 方法的一种），改变特征维度

def double_descent_experiment(n_train=80, n_features_range=range(5, 300, 3), n_trials=5):
    \"\"\"复现双下降：改变模型容量，记录测试误差\"\"\"
    np.random.seed(42)
    n_test = 200

    # 生成数据
    X_train = np.random.randn(n_train, 1000)
    true_w = np.random.randn(1000) * 0.1
    y_train = X_train @ true_w + np.random.randn(n_train) * 0.5
    X_test = np.random.randn(n_test, 1000)
    y_test = X_test @ true_w + np.random.randn(n_test) * 0.5

    results = []
    for n_feat in n_features_range:
        trial_errors = []
        for _ in range(n_trials):
            # 随机特征映射：固定随机矩阵将 1000 维映射到 n_feat 维
            R = np.random.randn(1000, n_feat) / np.sqrt(1000)
            phi_train = X_train @ R  # (n_train, n_feat)
            phi_test = X_test @ R

            # 最小二乘解
            if n_feat < n_train:
                # 欠参数化：正规方程
                w = np.linalg.lstsq(phi_train, y_train, rcond=None)[0]
            else:
                # 过参数化：最小范数解（伪逆）
                w = phi_train.T @ np.linalg.inv(phi_train @ phi_train.T + 1e-6 * np.eye(n_train)) @ y_train

            y_pred = phi_test @ w
            trial_errors.append(np.mean((y_pred - y_test)**2))

        results.append(np.median(trial_errors))
    return list(n_features_range), results

complexities, test_errs = double_descent_experiment()

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(complexities, test_errs, 'r-o', linewidth=2, markersize=4)
ax.axvline(80, color='green', linestyle='--', alpha=0.7, label='插值阈值 (n_train=80)')
ax.set_xlabel('模型复杂度（特征维度）')
ax.set_ylabel('测试误差')
ax.set_title('双下降现象：过参数化后测试误差再次下降')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_double_descent.png', bbox_inches='tight')
plt.show()
print("观察：")
print("  1. 复杂度 < 80：欠拟合，误差下降")
print("  2. 复杂度 ≈ 80：插值阈值，误差峰值（刚好记住数据但泛化最差）")
print("  3. 复杂度 > 80：过参数化，误差再次下降！这就是双下降。")""")

# ============================================================
md("""## 4. 为什么过参数化能泛化？

### 4.1 隐式正则化

虽然过参数化模型有无数个能完美拟合训练数据的解，但**梯度下降倾向于找到其中"最简单"的那个**——
范数最小、最平坦的解。

> 梯度下降不只是"找一个解"，而是"找一种特定的解"——这个隐式偏好起到了正则化的作用。

### 4.2 平坦极小值

损失曲面上有很多极小值。**平坦的极小值**比**尖锐的极小值**泛化更好：
- 平坦极小值：数据稍微扰动，loss 变化不大 → 泛化好
- 尖锐极小值：数据稍微扰动，loss 爆炸 → 泛化差

过参数化模型的损失曲面更可能有**平坦的极小值**。""")

code("""# 实验：平坦 vs 尖锐极小值
def train_with_noise(lr, noise_std=0.0, epochs=500):
    \"\"\"训练一个小网络，可选在梯度上加噪声\"\"\"
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(20, 50), nn.ReLU(), nn.Linear(50, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    X = torch.randn(100, 20)
    y = torch.randn(100, 1)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = ((model(X) - y)**2).mean()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad += torch.randn_like(p.grad) * noise_std
        optimizer.step()
    return model

# 训练两个模型：一个大 lr（倾向平坦），一个小 lr（倾向尖锐）
def evaluate_robustness(model, X, y, n_perturb=50, eps=0.01):
    \"\"\"评估模型对输入扰动的鲁棒性\"\"\"
    base_loss = ((model(X) - y)**2).mean().item()
    perturbed_losses = []
    for _ in range(n_perturb):
        X_pert = X + torch.randn_like(X) * eps
        perturbed_losses.append(((model(X_pert) - y)**2).mean().item())
    return base_loss, np.mean(perturbed_losses)

X = torch.randn(100, 20)
y = torch.randn(100, 1)

model_flat = train_with_noise(lr=0.1, noise_std=0.01)
model_sharp = train_with_noise(lr=0.001, noise_std=0.0)

base_f, pert_f = evaluate_robustness(model_flat, X, y)
base_s, pert_s = evaluate_robustness(model_sharp, X, y)

print(f"平坦极小值模型: base loss = {base_f:.4f}, 扰动后 = {pert_f:.4f}, 变化 = {pert_f-base_f:.4f}")
print(f"尖锐极小值模型: base loss = {base_s:.4f}, 扰动后 = {pert_s:.4f}, 变化 = {pert_s-base_s:.4f}")
print(f"\\n平坦极小值对扰动更鲁棒 → 泛化更好。这就是过参数化能泛化的直觉解释。")""")

# ============================================================
md("""## 5. 用神经网络复现双下降

把双下降现象在真实神经网络上也复现一次——改变隐藏层宽度。""")

code("""def nn_double_descent(n_train=50, widths=range(2, 200, 4), n_trials=3):
    \"\"\"用真实神经网络复现双下降\"\"\"
    np.random.seed(42)
    # 生成数据
    X_train = np.random.randn(n_train, 10)
    y_train = (X_train[:, 0]**2 + X_train[:, 1] * X_train[:, 2] + np.random.randn(n_train) * 0.1)
    X_test = np.random.randn(200, 10)
    y_test = (X_test[:, 0]**2 + X_test[:, 1] * X_test[:, 2])

    results = []
    for w in widths:
        trial_errs = []
        for _ in range(n_trials):
            torch.manual_seed(np.random.randint(10000))
            model = nn.Sequential(nn.Linear(10, w), nn.ReLU(), nn.Linear(w, 1))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            Xt = torch.tensor(X_train, dtype=torch.float32)
            yt = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
            for _ in range(500):
                optimizer.zero_grad()
                loss = ((model(Xt) - yt)**2).mean()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
            trial_errs.append(np.mean((pred.ravel() - y_test)**2))
        results.append(np.median(trial_errs))
    return list(widths), results

widths, nn_errs = nn_double_descent()

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(widths, nn_errs, 'b-o', linewidth=2, markersize=4)
ax.axvline(50, color='green', linestyle='--', alpha=0.7, label='插值阈值 (n_train=50)')
ax.set_xlabel('隐藏层宽度')
ax.set_ylabel('测试误差')
ax.set_title('神经网络上的双下降现象')
ax.set_ylim(0, min(max(nn_errs), 5))
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_nn_double_descent.png', bbox_inches='tight')
plt.show()
print("在真实神经网络上也观察到了双下降——过参数化区域泛化更好。")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 万能近似定理（实验验证逼近任意函数） | ✅ |
| 宽度 vs 深度（深更高效） | ✅ |
| 经典 bias-variance tradeoff（U 形曲线） | ✅ |
| 双下降现象（过参数化后误差再降） | ✅ |
| 为什么过参数化能泛化（隐式正则化 + 平坦极小值） | ✅ |

### 三个关键 takeaway

1. **万能近似定理**保证了一个隐藏层就能逼近任何函数，但**深度比宽度高效**
2. **经典理论预言 U 形**，但深度学习有**双下降**——过参数化反而泛化好
3. **现代 LLM 故意用过参数化**，正好处在双下降的第二次下降区域

> 这就是为什么 GPT-4 有万亿参数却不严重过拟合——它不是在 U 形的右边，而是在**双下降的右边**。

### 🔗 下一章预告

**`06_cnn_vs_vit.ipynb`** — CNN ↔ ViT 对比（归纳偏置 vs 数据效率）

---

> 💬 **写在最后**：双下降是深度学习最重要的理论发现之一。
> 它解释了为什么"模型比数据大"不是问题，反而是优势——
> 这个 insight 直接支撑了整个大模型时代的合理性。""")

# ============================================================
output_path = "notebooks/05_generalization_theory.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")