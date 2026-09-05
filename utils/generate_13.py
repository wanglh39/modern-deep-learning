"""生成 13_dpo_family.ipynb 的脚本"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "$schema": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

def md(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    nb.cells.append(nbf.v4.new_code_cell(text))

# ============================================================
md("""# 13 — 🔥 DPO 系列：直接偏好优化

> RLHF 用 PPO 训练——但不稳定、计算昂贵、需要4个模型。
> DPO 的洞察：RLHF 的最优解有**闭式形式**，可以跳过奖励模型直接优化策略。
> 这让对齐训练变得简单稳定——DPO 成了 2024 年的主流选择。

## 本章你将掌握

1. **DPO**：从 RLHF 推导出直接优化目标
2. **DPO 损失**：偏好分类的二元交叉熵
3. **GRPO**：组内相对优化（DeepSeek 用的）
4. **DAPO/GSPO**：DPO 的改进变体
5. **对比**：PPO vs DPO vs GRPO""")

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
md("""## 1. DPO：从 RLHF 到直接优化

### 1.1 DPO 的数学推导

RLHF 的目标：

$$\\max_\\pi E[r(x,y)] - \\beta KL(\\pi || \\pi_{ref})$$

这个优化有**闭式最优解**：

$$\\pi^*(y|x) = \\frac{1}{Z(x)} \\pi_{ref}(y|x) \\exp\\left(\\frac{r(x,y)}{\\beta}\\right)$$

反过来，奖励可以用策略表示：

$$r(x,y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{ref}(y|x)} + \\beta \\log Z(x)$$

代入偏好模型 $P(y_w > y_l) = \\sigma(r(y_w) - r(y_l))$，$Z(x)$ 消掉：

$$P(y_w > y_l) = \\sigma\\left(\\beta \\log \\frac{\\pi^*(y_w|x)}{\\pi_{ref}(y_w|x)} - \\beta \\log \\frac{\\pi^*(y_l|x)}{\\pi_{ref}(y_l|x)}\\right)$$

### 1.2 DPO 损失

直接最大化偏好概率：

$$L_{DPO} = -E\\left[\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{ref}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{ref}(y_l|x)}\\right)\\right]$$

```
RLHF:  偏好 → 奖励模型 → PPO → 策略    (3步, 4个模型)
DPO:   偏好 → 直接优化策略              (1步, 2个模型: π_θ + π_ref)
```

> 💡 DPO 把 RLHF 从强化学习问题变成了**分类问题**——用 BCE 损失训练，简单稳定。""")

code("""class TinyPolicy(nn.Module):
    \"\"\"简化策略: prompt → response 概率\"\"\"
    def __init__(self, vocab_size=10, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.fc = nn.Linear(d, vocab_size)
    def forward(self, x):
        return F.log_softmax(self.fc(self.embed(x).mean(dim=1)), dim=-1)

def dpo_loss(policy_logps, ref_logps, chosen_idx, rejected_idx, beta=0.1):
    \"\"\"DPO 损失
    policy_logps: (batch, vocab) 策略的 log 概率
    ref_logps: (batch, vocab) 参考模型的 log 概率
    chosen_idx: (batch,) chosen action 的 index
    rejected_idx: (batch,) rejected action 的 index
    \"\"\"
    # π_θ(y_w) / π_ref(y_w) 的 log
    policy_chosen = policy_logps.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
    policy_rejected = policy_logps.gather(1, rejected_idx.unsqueeze(1)).squeeze(1)
    ref_chosen = ref_logps.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)
    ref_rejected = ref_logps.gather(1, rejected_idx.unsqueeze(1)).squeeze(1)

    # logits = β * (log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l)))
    logits = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    # DPO 损失 = -log σ(logits) = BCE
    return -F.logsigmoid(logits).mean()

# 生成偏好数据
vocab_size = 8
n_samples = 100
prompts = torch.randint(0, vocab_size, (n_samples, 3))
# 假设: 偶数 token 是 "好" 的, 奇数是 "差" 的
chosen = torch.randint(0, vocab_size//2, (n_samples,)) * 2      # 偶数
rejected = torch.randint(0, vocab_size//2, (n_samples,)) * 2 + 1  # 奇数

# 参考模型 (SFT)
ref_model = TinyPolicy(vocab_size)
# DPO 模型
dpo_model = TinyPolicy(vocab_size)
dpo_model.load_state_dict(ref_model.state_dict())

optimizer = torch.optim.Adam(dpo_model.parameters(), lr=1e-2)
losses_dpo = []

for epoch in range(200):
    logps = dpo_model(prompts)
    with torch.no_grad():
        ref_logps = ref_model(prompts)
    loss = dpo_loss(logps, ref_logps, chosen, rejected, beta=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses_dpo.append(loss.item())

print(f"DPO 训练完成, 最终损失: {losses_dpo[-1]:.4f}")

# 验证: chosen 的概率应该比 rejected 高
with torch.no_grad():
    logps = dpo_model(prompts[:5])
    ref_logps = ref_model(prompts[:5])
    for i in range(5):
        c_prob = logps[i, chosen[i]].exp().item()
        r_prob = logps[i, rejected[i]].exp().item()
        print(f"  样本{i}: chosen概率={c_prob:.3f}, rejected概率={r_prob:.3f}, 比率={c_prob/r_prob:.2f}")""")

# ============================================================
md("""## 2. DPO vs PPO 对比

### 2.1 架构对比

| | PPO (RLHF) | DPO |
|---|------------|-----|
| **模型数** | 4 (policy+value+reward+ref) | 2 (policy+ref) |
| **训练类型** | 强化学习 | 分类 (BCE) |
| **稳定性** | ❌ 敏感 | ✅ 稳定 |
| **采样** | ✅ 需要在线采样 | ❌ 离线即可 |
| **计算量** | 高 | 低 |

### 2.2 为什么 DPO 更简单？

```
PPO:  生成 → 奖励 → 优势 → PPO更新 → KL惩罚    (复杂管线)
DPO:  偏好数据 → BCE损失 → 梯度更新              (一步到位)
```

> 💡 DPO 不需要在线采样——直接用偏好数据训练。这让训练管线简化 10x。""")

code("""# 对比 DPO 和 PPO 的训练曲线
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# DPO 曲线
axes[0].plot(losses_dpo, 'b-', linewidth=2, label='DPO')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('DPO Loss')
axes[0].set_title('DPO: 训练损失 (分类问题)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 模拟 PPO 曲线 (更不稳定)
np.random.seed(42)
ppo_rewards = np.cumsum(np.random.randn(200) * 0.1 + 0.05) + 0.3
ppo_rewards = ppo_rewards - ppo_rewards.min() + 0.1
# 加一些不稳定
ppo_rewards[50:55] -= 0.3
ppo_rewards[120:125] += 0.2

axes[1].plot(ppo_rewards, 'r-', linewidth=2, label='PPO (模拟)')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Reward')
axes[1].set_title('PPO: 奖励曲线 (有波动)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_dpo_vs_ppo.png', bbox_inches='tight')
plt.show()
print("DPO: 损失平稳下降 (分类问题); PPO: 奖励有波动 (RL问题)。")""")

# ============================================================
md("""## 3. GRPO：组相对策略优化

### 3.1 GRPO 的思想

GRPO（DeepSeek 用的）去掉价值模型，用**组内相对**估计优势：

对每个 prompt 生成 $G$ 个回答，用组内奖励排序估计优势：

$$A_i = \\frac{r_i - \\text{mean}(r_1...r_G)}{\\text{std}(r_1...r_G)}$$

```
PPO:  每个prompt生成1个回答, 用价值模型估计优势    (需要价值模型)
GRPO: 每个prompt生成G个回答, 用组内奖励排序估计优势  (不需要价值模型!)
```

### 3.2 GRPO 损失

$$L_{GRPO} = -E\\left[\\frac{1}{G}\\sum_i \\min(r_i A_i, \\text{clip}(r_i) A_i) - \\beta KL\\right]$$

和 PPO 类似但优势用组内相对计算，不需要价值网络。

> 💡 GRPO 是 DeepSeek-R1 的核心——去掉价值模型，训练更简单更省显存。""")

code("""def grpo_step(policy, ref_model, prompt, reward_fn, group_size=4, beta=0.1, clip_eps=0.2):
    \"\"\"单步 GRPO 更新\"\"\"
    # 1. 生成 G 个回答
    with torch.no_grad():
        logps_all = policy(prompt.expand(group_size, -1))
        actions = torch.multinomial(logps_all.exp(), 1).squeeze()
        ref_logps_all = ref_model(prompt.expand(group_size, -1))

    # 2. 计算奖励
    rewards = torch.tensor([reward_fn(a.item()) for a in actions], dtype=torch.float32)

    # 3. 组内相对优势 (不需要价值模型!)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 4. PPO 式更新
    policy_logps = policy(prompt.expand(group_size, -1))
    log_ratio = policy_logps.gather(1, actions.unsqueeze(1)).squeeze(1) - \
                logps_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    ratio = log_ratio.exp()

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL 惩罚
    kl = (logps_all.exp() * (logps_all - ref_logps_all)).sum(dim=-1).mean()

    return policy_loss + beta * kl, rewards.mean().item()

# GRPO 训练
grpo_model = TinyPolicy(vocab_size)
grpo_model.load_state_dict(ref_model.state_dict())
optimizer = torch.optim.Adam(grpo_model.parameters(), lr=1e-2)

# 奖励函数: 偶数 token 奖励高
def reward_fn(action):
    return 1.0 if action % 2 == 0 else -0.5

grpo_rewards = []
for step in range(200):
    prompt = torch.randint(0, vocab_size, (1, 3))
    loss, avg_reward = grpo_step(grpo_model, ref_model, prompt, reward_fn, group_size=8, beta=0.05)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    grpo_rewards.append(avg_reward)

print(f"GRPO 训练完成")
print(f"  初始平均奖励: {np.mean(grpo_rewards[:20]):.3f}")
print(f"  最终平均奖励: {np.mean(grpo_rewards[-20:]):.3f}")
print(f"  GRPO 不需要价值模型——用组内排序估计优势")""")

# ============================================================
md("""## 4. DPO 变体：DAPO / GSPO

### 4.1 DPO 的问题

DPO 虽好但有缺陷：
1. **长度偏置**：DPO 偏好短回答（log 概率对长序列更低）
2. **分布偏移**：离线数据可能和当前策略差很远

### 4.2 DAPO (Decoupled Alignment PO)

解耦 chosen 和 rejected 的 clip，分别处理：

$$L_{DAPO} = -\\log \\sigma(\\beta \\log \\frac{\\pi_\\theta(y_w)}{\\pi_{ref}(y_w)} \\cdot w_w - \\beta \\log \\frac{\\pi_\\theta(y_l)}{\\pi_{ref}(y_l)} \\cdot w_l)$$

其中 $w_w, w_l$ 是解耦的权重，缓解长度偏置。

### 4.3 GSPO (Group Sorted PO)

用组内排序替代 sigmoid，更鲁棒：

$$L_{GSPO} = -\\log \\sigma\\left(\\beta \\log \\frac{\\pi_\\theta(y_w)}{\\pi_\\theta(y_l)} - \\beta \\log \\frac{\\pi_{ref}(y_w)}{\\pi_{ref}(y_l)}\\right)$$

> 💡 这些变体都是 2024-2025 年的新方法，针对 DPO 的具体缺陷改进。""")

code("""# 对比不同 DPO 变体的损失
def standard_dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    \"\"\"标准 DPO\"\"\"
    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    return -F.logsigmoid(logits)

def length_normalized_dpo(pi_c, pi_r, ref_c, ref_r, len_c=1, len_r=1, beta=0.1):
    \"\"\"长度归一化 DPO (缓解长度偏置)\"\"\"
    logits = beta * ((pi_c/len_c - ref_c/len_c) - (pi_r/len_r - ref_r/len_r))
    return -F.logsigmoid(logits)

def gspo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    \"\"\"GSPO: 用比率而非差值\"\"\"
    logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
    return -F.logsigmoid(logits)

# 模拟不同长度回答的损失
fig, ax = plt.subplots(figsize=(10, 6))
lengths = range(1, 51)
for label, loss_fn, color in [
    ('标准DPO', standard_dpo_loss, 'blue'),
    ('长度归一DPO', length_normalized_dpo, 'green'),
    ('GSPO', gspo_loss, 'red'),
]:
    losses = []
    for L in lengths:
        # chosen 是短回答 (长度1), rejected 是长回答 (长度L)
        pi_c = torch.tensor(-0.5)  # 短回答 log prob
        pi_r = torch.tensor(-0.5 * L)  # 长回答 log prob (更负)
        ref_c = torch.tensor(-0.5)
        ref_r = torch.tensor(-0.5 * L)
        if label == '长度归一DPO':
            loss = loss_fn(pi_c, pi_r, ref_c, ref_r, len_c=1, len_r=L)
        else:
            loss = loss_fn(pi_c, pi_r, ref_c, ref_r)
        losses.append(loss.item())
    ax.plot(lengths, losses, color=color, linewidth=2.5, label=label)

ax.set_xlabel('Rejected 回答长度')
ax.set_ylabel('损失')
ax.set_title('DPO 变体: 长度偏置对比')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_dpo_variants.png', bbox_inches='tight')
plt.show()
print("标准DPO 随 rejected 长度增加损失增大 (长度偏置); 归一化版本更稳定。")""")

# ============================================================
md("""## 5. 对比总结：PPO vs DPO vs GRPO

| 方法 | 模型数 | 训练类型 | 稳定性 | 在线采样 | 代表应用 |
|------|--------|---------|--------|---------|---------|
| **PPO** | 4 | RL | ⚠️ | ✅ | ChatGPT |
| **DPO** | 2 | 分类 | ✅ | ❌ | Llama-2/3 |
| **GRPO** | 2 | RL (无价值) | ✅ | ✅ | DeepSeek-R1 |
| **DAPO** | 2 | 分类 | ✅ | ❌ | 2024 新方法 |
| **GSPO** | 2 | 分类 | ✅ | ❌ | 2024 新方法 |

### 选择指南

```
有在线采样能力 + 追求最优 → PPO (经典但复杂)
有偏好数据 + 想简单稳定   → DPO (主流选择)
有奖励函数 + 无价值模型   → GRPO (DeepSeek 路线)
DPO 有长度偏置问题        → DAPO/GSPO
```

> 💡 2024 年趋势：**DPO 系列正在取代 PPO** 成为主流对齐方法。
> DeepSeek-R1 用 GRPO 证明了无价值模型的 RL 也能很强。""")

code("""# 方法对比可视化
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

methods = ['PPO', 'DPO', 'GRPO', 'DAPO', 'GSPO']
models_count = [4, 2, 2, 2, 2]
stability = [2, 5, 4, 5, 5]  # 1-5 评分
colors = ['red', 'blue', 'green', 'orange', 'purple']

axes[0].bar(methods, models_count, color=colors, alpha=0.7)
axes[0].set_ylabel('需要的模型数')
axes[0].set_title('模型数对比 (越少越好)')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(methods, stability, color=colors, alpha=0.7)
axes[1].set_ylabel('训练稳定性 (1-5)')
axes[1].set_title('稳定性对比 (越高越好)')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, 6)

plt.tight_layout()
plt.savefig('notebooks/fig_alignment_methods.png', bbox_inches='tight')
plt.show()
print("DPO 系列用更少模型达到更高稳定性——这是它取代 PPO 的原因。")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| DPO 从 RLHF 推导出闭式解 | ✅ |
| DPO 损失 = 偏好分类 BCE | ✅ |
| DPO vs PPO 的架构对比 | ✅ |
| GRPO 组内相对优势 (无价值模型) | ✅ |
| DAPO/GSPO 解决 DPO 长度偏置 | ✅ |

### 核心 takeaway

> **DPO 把对齐从强化学习变成分类**——简单、稳定、省资源。
> GRPO 去掉价值模型，DAPO/GSPO 修复长度偏置。
> 2024 年 DPO 系列正在取代 PPO 成为主流。

### 🔗 下一章预告

**`14_data_engineering.ipynb`** — 合成数据、筛选、配比 + 自训练/自我改进

---

> 💬 **写在最后**：DPO 的美在于数学——RLHF 的最优解有闭式形式，
> 让我们跳过奖励模型直接优化策略。这是"化繁为简"的典范。""")

# ============================================================
output_path = "notebooks/13_dpo_family.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")