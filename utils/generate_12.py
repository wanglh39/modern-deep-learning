"""生成 12_rlhf_ppo.ipynb 的脚本"""
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
md("""# 12 — 🧭 RLHF + PPO：从人类偏好学习

> 预训练让模型会说话，SFT 让模型听指令，但模型不知道什么是"好"。
> RLHF（Reinforcement Learning from Human Feedback）用**人类偏好**作为奖励信号，
> 通过强化学习让模型生成人类喜欢的回答。这是 ChatGPT 的关键一步。

## 本章你将掌握

1. **RLHF 三阶段**：SFT → 奖励模型 → PPO 训练
2. **奖励模型**：从偏好数据学习人类偏好
3. **PPO 算法**：策略梯度 + 裁剪 + 价值函数
4. **KL 散度惩罚**：防止策略偏离参考模型太远
5. **完整 RLHF 循环**：在小任务上实现端到端""")

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
md("""## 1. RLHF 三阶段流程

```
阶段1: 预训练 (CLM)     →  会生成文本的基座模型
阶段2: SFT (指令微调)    →  能跟随指令的模型
阶段3: RLHF:
  3a. 训练奖励模型       →  学会打分 (人类偏好 → 标量奖励)
  3b. PPO 训练           →  用奖励信号优化策略 (生成模型)
  3c. KL 惩罚            →  防止偏离太远 (保持多样性)
```

### 为什么需要 RLHF？

SFT 只学了"指令→回答"的映射，但不知道回答的**质量**。
RLHF 通过人类偏好（"回答A比回答B好"）学一个奖励函数，再用 RL 优化：

| | SFT | RLHF |
|---|------|------|
| **信号** | 固定答案 | 偏好排序 |
| **优化** | 交叉熵 | 奖励最大化 |
| **效果** | 会回答 | 回答得好 |

> 💡 RLHF 的核心洞察：**人类偏好比标注答案更容易获取**——比较两个回答比写一个好回答容易。""")

# ============================================================
md("""## 2. 奖励模型

### 2.1 从偏好到奖励

人类标注员比较两个回答：给定 prompt，回答 A vs 回答 B，选更好的。
偏好数据 $\\{(prompt, y_w, y_l)\\}$，$y_w$ 是更好的，$y_l$ 是更差的。

奖励模型学的是 Bradley-Terry 模型：

$$P(y_w > y_l | x) = \\sigma(r(x, y_w) - r(x, y_l))$$

损失函数（最大化偏好概率）：

$$L = -\\log \\sigma(r(x, y_w) - r(x, y_l))$$

### 2.2 奖励模型架构

奖励模型通常用 SFT 模型改造——把 LM head 换成**标量输出**（奖励值）。""")

code("""class SimpleRewardModel(nn.Module):
    \"\"\"简化奖励模型: 文本 → 标量奖励\"\"\"
    def __init__(self, vocab_size=20, d_model=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, 1)  # 标量输出

    def forward(self, x):
        \"\"\"x: (batch, seq_len) token ids → (batch,) 奖励值\"\"\"
        h = self.embed(x).mean(dim=1)  # 平均池化
        h = F.relu(self.fc1(h))
        return self.fc2(h).squeeze(-1)  # (batch,)

# 生成模拟偏好数据
# 假设: "good" 开头的回答更好, "bad" 开头的更差
vocab = ['the', 'cat', 'sat', 'good', 'bad', 'helpful', 'wrong', 'clear',
         'confusing', 'right', 'nice', 'terrible', 'great', 'awful', 'on', 'mat']
word2id = {w: i for i, w in enumerate(vocab)}

# 偏好数据: (prompt, chosen, rejected)
preference_data = [
    ([word2id['the']], [word2id['good'], word2id['helpful']], [word2id['bad'], word2id['wrong']]),
    ([word2id['the']], [word2id['clear'], word2id['right']], [word2id['confusing'], word2id['awful']]),
    ([word2id['the']], [word2id['great'], word2id['nice']], [word2id['terrible'], word2id['bad']]),
    ([word2id['the']], [word2id['good'], word2id['clear']], [word2id['bad'], word2id['confusing']]),
    ([word2id['the']], [word2id['helpful'], word2id['right']], [word2id['wrong'], word2id['awful']]),
]

# 训练奖励模型
reward_model = SimpleRewardModel(vocab_size=len(vocab))
optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-2)
losses_rm = []

for epoch in range(200):
    total_loss = 0
    for prompt, chosen, rejected in preference_data:
        p = torch.tensor([prompt])
        c = torch.tensor([chosen])
        r = torch.tensor([rejected])
        r_chosen = reward_model(c)
        r_rejected = reward_model(r)
        # Bradley-Terry 损失: -log σ(r_chosen - r_rejected)
        loss = -F.logsigmoid(r_chosen - r_rejected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses_rm.append(total_loss / len(preference_data))

print(f"奖励模型训练完成, 最终损失: {losses_rm[-1]:.4f}")

# 验证奖励模型
test_responses = {
    'good helpful': [word2id['good'], word2id['helpful']],
    'bad wrong': [word2id['bad'], word2id['wrong']],
    'clear right': [word2id['clear'], word2id['right']],
    'confusing awful': [word2id['confusing'], word2id['awful']],
}
print("\\n奖励模型打分:")
for name, tokens in test_responses.items():
    with torch.no_grad():
        reward = reward_model(torch.tensor([tokens])).item()
    print(f"  '{name}': reward = {reward:.3f}")""")

# ============================================================
md("""## 3. PPO：近端策略优化

### 3.1 策略梯度回顾

RL 的目标是最大化期望奖励：

$$J(\\theta) = E_{a \\sim \\pi_\\theta}[R(a)]$$

策略梯度：

$$\\nabla J = E[R(a) \\nabla \\log \\pi_\\theta(a)]$$

### 3.2 PPO 的核心：裁剪

PPO 用**裁剪**限制策略更新幅度，防止一步更新太大：

$$L^{PPO} = E[\\min(r_t A_t, \\text{clip}(r_t, 1-\\epsilon, 1+\\epsilon) A_t)]$$

其中 $r_t = \\frac{\\pi_\\theta(a|s)}{\\pi_{old}(a|s)}$ 是重要性比率，$A_t$ 是优势函数。

```
不裁剪:  策略可能一步跳太远 → 不稳定
PPO裁剪: 把比率 r_t 限制在 [1-ε, 1+ε] → 稳定更新
         ε=0.2 是常用值
```

### 3.3 PPO 的完整损失

$$L = L^{PPO} - c_1 L^{VF} + c_2 S[\\pi]$$

- $L^{VF}$：价值函数损失（减少奖励估计方差）
- $S[\\pi]$：熵奖励（鼓励探索）
- $c_1, c_2$：系数""")

code("""class SimplePolicy(nn.Module):
    \"\"\"简化策略网络: 状态 → 动作概率\"\"\"
    def __init__(self, state_dim, action_dim, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_dim)

    def forward(self, x):
        return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=-1)

    def log_prob(self, x, action):
        probs = self.forward(x)
        return torch.log(probs.gather(1, action.unsqueeze(1)) + 1e-8).squeeze(1)

class SimpleValue(nn.Module):
    \"\"\"价值网络: 状态 → 价值估计\"\"\"
    def __init__(self, state_dim, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)

# PPO 超参数
clip_epsilon = 0.2
gamma = 0.99       # 折扣因子
lam = 0.95         # GAE 参数
lr_policy = 3e-3
lr_value = 1e-2
n_epochs = 10      # 每批数据训练几轮
n_updates = 100    # 总更新次数

state_dim = 4
action_dim = 3

policy = SimplePolicy(state_dim, action_dim)
value_fn = SimpleValue(state_dim)
opt_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy)
opt_value = torch.optim.Adam(value_fn.parameters(), lr=lr_value)

# 简单环境: 状态 → 动作 → 奖励 (用预定义的奖励函数模拟)
def simple_env_reward(state, action):
    \"\"\"模拟奖励: 某个动作在某些状态下更好\"\"\"
    return (state * torch.tensor([1.0, -0.5, 0.3, 0.8])).sum() * (action - 1) * -0.5 + 0.1 * action

print(f"PPO 设置: state_dim={state_dim}, action_dim={action_dim}, clip={clip_epsilon}")
print("策略网络和价值网络初始化完成")""")

code("""# PPO 训练循环
def collect_trajectory(policy, value_fn, n_steps=64):
    \"\"\"收集一条轨迹\"\"\"
    states, actions, rewards, log_probs, values = [], [], [], [], []
    for _ in range(n_steps):
        state = torch.randn(1, state_dim)
        with torch.no_grad():
            probs = policy(state)
            action = torch.multinomial(probs, 1).squeeze()
            log_p = torch.log(probs[0, action] + 1e-8)
            v = value_fn(state)
        reward = simple_env_reward(state.squeeze(), action)

        states.append(state.squeeze())
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_p)
        values.append(v)

    return torch.stack(states), torch.stack(actions), torch.tensor(rewards), torch.stack(log_probs), torch.stack(values).squeeze()

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    \"\"\"广义优势估计 (GAE)\"\"\"
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t+1] if t+1 < len(values) else 0) - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns

# 训练 PPO
mean_rewards = []
for update in range(n_updates):
    # 1. 收集数据
    states, actions, rewards, old_log_probs, values = collect_trajectory(policy, value_fn)
    advantages, returns = compute_gae(rewards, values, gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 2. PPO 更新 (多轮)
    for _ in range(n_epochs):
        # 策略更新
        new_log_probs = policy.log_prob(states, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 熵奖励 (鼓励探索)
        with torch.no_grad():
            probs = policy(states)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        policy_loss -= 0.01 * entropy

        opt_policy.zero_grad()
        policy_loss.backward()
        opt_policy.step()

        # 价值更新
        v_pred = value_fn(states)
        value_loss = F.mse_loss(v_pred, returns)
        opt_value.zero_grad()
        value_loss.backward()
        opt_value.step()

    mean_rewards.append(rewards.mean().item())
    if (update + 1) % 20 == 0:
        print(f"Update {update+1:3d}: 平均奖励={rewards.mean().item():.3f}, 策略损失={policy_loss.item():.4f}")

print(f"\\nPPO 训练完成, 最终平均奖励: {mean_rewards[-1]:.3f}")""")

code("""# 可视化 PPO 训练曲线
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(mean_rewards, 'b-', linewidth=2)
axes[0].set_xlabel('Update')
axes[0].set_ylabel('平均奖励')
axes[0].set_title('PPO 训练: 奖励曲线')
axes[0].grid(True, alpha=0.3)

# 可视化策略
with torch.no_grad():
    test_states = torch.randn(100, state_dim)
    probs = policy(test_states)
    actions = probs.argmax(dim=-1).numpy()

axes[1].hist(actions, bins=range(action_dim+1), align='left', rwidth=0.8, color='steelblue', alpha=0.7)
axes[1].set_xticks(range(action_dim))
axes[1].set_xlabel('动作')
axes[1].set_ylabel('选择次数')
axes[1].set_title('PPO 训练后: 策略动作分布')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/fig_ppo_training.png', bbox_inches='tight')
plt.show()
print("PPO 通过裁剪保证稳定更新——这是它比普通策略梯度更可靠的原因。")""")

# ============================================================
md("""## 4. KL 散度惩罚：防止偏离

### 4.1 为什么需要 KL 惩罚？

RLHF 优化奖励时，策略可能**走捷径**——找到奖励模型的漏洞而非真正变好。
解法：加 KL 散度惩罚，限制策略不偏离参考模型（SFT 模型）太远：

$$L = E[r(x,y)] - \\beta \\cdot KL(\\pi_\\theta || \\pi_{ref})$$

```
无 KL:  策略可能找到奖励漏洞 → 奖励高但回答差 (reward hacking)
有 KL:  策略在追求奖励的同时保持接近 SFT → 安全优化
```

### 4.2 β 的作用

$\\beta$ 控制 KL 惩罚的强度：
- $\\beta$ 太小 → 不够约束，可能 reward hacking
- $\\beta$ 太大 → 过度约束，学不动
- 典型值：0.01-0.1""")

code("""# 演示 KL 惩罚的效果
# 模拟: 策略概率 vs 参考概率
ref_probs = torch.tensor([0.3, 0.4, 0.3])  # 参考模型 (SFT)
# 不同 β 下的最优策略
betas = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
# 假设奖励: 动作0奖励高 (模拟 reward hacking)
rewards = torch.tensor([1.0, 0.0, 0.0])

fig, ax = plt.subplots(figsize=(10, 6))
for i, beta in enumerate(betas):
    # 最优策略: 最大化 r - β * KL(π || ref)
    # 简化: 用网格搜索
    best_prob = None
    best_obj = -float('inf')
    for p0 in np.linspace(0.01, 0.98, 50):
        for p1 in np.linspace(0.01, 0.98 - p0, 50):
            p2 = 1 - p0 - p1
            if p2 <= 0: continue
            p = torch.tensor([p0, p1, p2])
            kl = (p * (torch.log(p) - torch.log(ref_probs))).sum().item()
            obj = (p * rewards).sum().item() - beta * kl
            if obj > best_obj:
                best_obj = obj
                best_prob = p.numpy()

    ax.bar(np.arange(3) + i*0.13, best_prob, width=0.12, label=f'β={beta}', alpha=0.8)

ax.set_xticks(np.arange(3) + 0.3)
ax.set_xticklabels(['动作0 (高奖励)', '动作1', '动作2'])
ax.set_ylabel('策略概率')
ax.set_title('KL 惩罚强度对策略的影响')
ax.legend(ncol=3, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('notebooks/fig_kl_penalty.png', bbox_inches='tight')
plt.show()
print("β=0 时策略全押高奖励动作 (reward hacking); β大时策略接近参考模型 (安全)。")""")

# ============================================================
md("""## 5. 完整 RLHF 流程

### 5.1 端到端流程

```
1. 预训练模型 π_pre (CLM)
2. SFT 模型 π_sft (指令微调)
3. 训练奖励模型 r(x,y) (偏好数据)
4. PPO 训练:
   for each iteration:
     a. 用 π_θ 生成回答 y
     b. 用 r(x,y) 打分
     c. 计算 KL(π_θ || π_sft)
     d. PPO 更新: 最大化 r - β*KL
5. 得到对齐模型 π_rlhf
```

### 5.2 RLHF 的挑战

| 挑战 | 说明 |
|------|------|
| **奖励模型不完美** | 偏好数据有噪声，模型可能学偏 |
| **Reward hacking** | 策略找漏洞而非真正变好 |
| **训练不稳定** | PPO 对超参数敏感 |
| **计算昂贵** | 每步要生成 + 4个模型前向 |

> 💡 这些挑战催生了 DPO（Direct Preference Optimization）——跳过奖励模型直接优化策略。
> 下一章讲 DPO 系列。""")

code("""# 简化完整 RLHF 演示: 在文本生成任务上
# 模拟: 生成模型生成回答, 奖励模型打分, PPO 优化
torch.manual_seed(42)

class TinyGenerator(nn.Module):
    \"\"\"简化生成模型: prompt → response 概率\"\"\"
    def __init__(self, vocab_size=5, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.fc = nn.Linear(d, vocab_size)
    def forward(self, x):
        return F.softmax(self.fc(self.embed(x).mean(dim=1)), dim=-1)

# 模拟环境: 某些 token 组合奖励高
good_tokens = {0, 1}  # "good" tokens
def get_reward(action):
    return 1.0 if action in good_tokens else -0.5

# 初始策略 (SFT 模型)
sft_model = TinyGenerator()
# RLHF 模型 (要训练的)
rlhf_model = TinyGenerator()
rlhf_model.load_state_dict(sft_model.state_dict())

beta = 0.1  # KL 惩罚系数
optimizer = torch.optim.Adam(rlhf_model.parameters(), lr=1e-2)

rewards_history = []
kl_history = []

for step in range(200):
    # 1. 生成: 用随机 prompt
    prompt = torch.tensor([[np.random.randint(0, 5)]])
    with torch.no_grad():
        sft_probs = sft_model(prompt)
    probs = rlhf_model(prompt)

    # 2. 采样动作
    action = torch.multinomial(probs, 1).squeeze()
    reward = get_reward(action.item())

    # 3. KL 散度
    kl = (probs * (torch.log(probs + 1e-8) - torch.log(sft_probs + 1e-8))).sum()

    # 4. PPO 式更新 (简化: 直接策略梯度)
    log_prob = torch.log(probs[0, action] + 1e-8)
    loss = -(reward * log_prob - beta * kl)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    rewards_history.append(reward)
    kl_history.append(kl.item())

print(f"RLHF 训练完成")
print(f"  平均奖励: {np.mean(rewards_history[-50:]):.3f} (初始: {np.mean(rewards_history[:50]):.3f})")
print(f"  KL 散度: {np.mean(kl_history[-50:]):.4f}")""")

code("""# 可视化 RLHF 训练
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 奖励曲线 (滑动平均)
window = 20
smoothed_r = [np.mean(rewards_history[max(0,i-window):i+1]) for i in range(len(rewards_history))]
axes[0].plot(smoothed_r, 'b-', linewidth=2)
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Step')
axes[0].set_ylabel('平均奖励 (滑动窗口)')
axes[0].set_title('RLHF: 奖励曲线')
axes[0].grid(True, alpha=0.3)

# KL 散度
smoothed_kl = [np.mean(kl_history[max(0,i-window):i+1]) for i in range(len(kl_history))]
axes[1].plot(smoothed_kl, 'r-', linewidth=2)
axes[1].set_xlabel('Step')
axes[1].set_ylabel('KL(π_θ || π_sft)')
axes[1].set_title('RLHF: KL 散度 (策略偏离)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_rlhf_full.png', bbox_inches='tight')
plt.show()
print("RLHF: 奖励上升, KL 散度控制在合理范围——β 平衡了奖励和偏离。")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| RLHF 三阶段流程 | ✅ |
| 奖励模型 (Bradley-Terry) | ✅ |
| PPO 裁剪机制 | ✅ |
| GAE 广义优势估计 | ✅ |
| KL 散度惩罚防偏离 | ✅ |
| 完整 RLHF 训练循环 | ✅ |

### 核心 takeaway

> **RLHF = 奖励模型 + PPO + KL 惩罚**。
> 人类偏好 → 奖励信号 → 强化学习优化 → 对齐人类意图。
> 但 RLHF 训练不稳定、计算昂贵——这催生了 DPO 系列方法。

### 🔗 下一章预告

**`13_dpo_family.ipynb`** — 🔥DPO → GRPO → DAPO/GSPO（跳过奖励模型的直接偏好优化）

---

> 💬 **写在最后**：RLHF 是 ChatGPT 的秘密武器——它让模型从"会说话"变成"说得好"。
> 理解 PPO 和 KL 惩罚，就理解了现代 LLM 对齐的基石。""")

# ============================================================
output_path = "notebooks/12_rlhf_ppo.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")