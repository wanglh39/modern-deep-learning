# 生成 72_deep_rl_frontiers.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 72 — 深度强化学习前沿：AlphaGo 与多智能体 RL

> 🧭🪙 从围棋到星际争霸，从单智能体到多智能体博弈——深度 RL 的前沿。

## 本章你将掌握

1. **AlphaGo/AlphaZero**：自我对弈 + MCTS
2. **多智能体 RL**：协作与竞争
3. **博弈论视角**：纳什均衡
4. **前沿方向**：AlphaStar、OpenAI Five""")

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

md("""## 1. AlphaGo 系列

### 1.1 从 AlphaGo 到 AlphaZero

```
AlphaGo (2016):
  - 监督学习 (人类棋谱) + RL + MCTS
  - 击败李世石

AlphaGo Zero (2017):
  - 纯自我对弈 (不用人类棋谱)
  - Tabula rasa (白板)
  - 击败 AlphaGo

AlphaZero (2017):
  - 通用于围棋/国际象棋/将棋
  - 同一算法，不同游戏

MuZero (2019):
  - 不需要游戏规则
  - 学习隐式状态模型
  - 适用于 Atari 游戏
```

### 1.2 核心算法

```
AlphaZero = 自我对弈 + MCTS + 神经网络

1. 神经网络:
   - 策略网络: p(a|s) → 建议走法
   - 价值网络: v(s) → 评估局面

2. MCTS (蒙特卡洛树搜索):
   - 用神经网络指导搜索
   - 选择 → 扩展 → 评估 → 回传

3. 自我对弈:
   - 用 MCTS 下棋
   - 生成训练数据
   - 训练网络
   - 重复
```

> 💡 AlphaZero 的革命：不需要人类知识，纯自我对弈达到超人类水平。""")

code("""# AlphaZero 简化: MCTS
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

    def ucb(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = c * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)
        return exploit + explore

class MCTS:
    def __init__(self, n_simulations=100):
        self.n_sim = n_simulations

    def search(self, root_state, policy_fn, value_fn):
        root = MCTSNode(root_state)

        for _ in range(self.n_sim):
            node = root
            state = root_state

            # 选择
            while node.children:
                action = max(node.children, key=lambda a: node.children[a].ucb())
                node = node.children[action]
                state = self._step(state, action)

            # 扩展
            policy = policy_fn(state)
            for action, prob in enumerate(policy):
                if prob > 0:
                    next_state = self._step(state, action)
                    child = MCTSNode(next_state, node)
                    child.prior = prob
                    node.children[action] = child

            # 评估
            value = value_fn(state)

            # 回传
            while node:
                node.visits += 1
                node.value += value
                value = -value  # 对手视角
                node = node.parent

        # 返回访问次数最多的动作
        best_action = max(root.children, key=lambda a: root.children[a].visits)
        return best_action, root

    def _step(self, state, action):
        return state + action  # 简化

# 模拟策略和价值函数
def policy_fn(state):
    probs = np.random.dirichlet(np.ones(5))
    return probs

def value_fn(state):
    return np.random.uniform(-1, 1)

# 运行 MCTS
mcts = MCTS(n_simulations=50)
best_action, root = mcts.search(0, policy_fn, value_fn)

print(f"MCTS 搜索:")
print(f"  模拟次数: {mcts.n_sim}")
print(f"  最佳动作: {best_action}")
print(f"  根节点访问: {root.visits}")
print(f"  子节点数: {len(root.children)}")""")

md("""## 2. 多智能体 RL

### 2.1 协作 vs 竞争

```
多智能体 RL (MARL):
  多个智能体在共享环境中交互

1. 纯协作 (Cooperative):
   所有智能体共享奖励
   → 一起最大化
   例: 多机器人搬运

2. 纯竞争 (Competitive):
   零和博弈
   一方赢 = 另一方输
   例: 围棋、星际争霸

3. 混合 (Mixed):
   既有合作又有竞争
   例: 队伍对抗 (5v5)
```

### 2.2 挑战

```
MARL 额外挑战:
  1. 非平稳性: 对手也在学习 → 环境非平稳
  2. 信用分配: 谁的贡献大?
  3. 可扩展性: 智能体多 → 维度爆炸
  4. 均衡选择: 多个均衡 → 选哪个?
```""")

code("""# 多智能体 RL 简化
class MultiAgentEnv:
    def __init__(self, n_agents=2):
        self.n_agents = n_agents
        self.state = np.zeros(n_agents)

    def step(self, actions):
        # 简化: 协作任务
        rewards = []
        for i in range(self.n_agents):
            r = -abs(actions[i] - self.state[i])
            rewards.append(r)
        self.state += np.array(actions) * 0.1
        team_reward = sum(rewards) / self.n_agents
        return self.state.copy(), [team_reward] * self.n_agents

class QAgent:
    def __init__(self, n_actions=5, lr=0.1, epsilon=0.1):
        self.q = np.zeros(n_actions)
        self.lr = lr
        self.epsilon = epsilon
        self.n_actions = n_actions

    def act(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q)

    def update(self, action, reward):
        self.q[action] += self.lr * (reward - self.q[action])

# 训练多智能体
env = MultiAgentEnv(n_agents=2)
agents = [QAgent() for _ in range(2)]

rewards_history = []
for episode in range(200):
    actions = [agent.act() for agent in agents]
    _, rewards = env.step(actions)
    for agent, action, reward in zip(agents, actions, rewards):
        agent.update(action, reward)
    rewards_history.append(np.mean(rewards))

print(f"多智能体 RL:")
print(f"  智能体数: {env.n_agents}")
print(f"  初始奖励: {rewards_history[0]:.4f}")
print(f"  最终奖励: {rewards_history[-1]:.4f}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rewards_history, 'b-', alpha=0.5)
window = 20
smooth = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
ax.plot(smooth, 'r-', linewidth=2, label='滑动平均')
ax.set_xlabel('回合', fontsize=12)
ax.set_ylabel('平均奖励', fontsize=12)
ax.set_title('多智能体协作 RL', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_marl.png', bbox_inches='tight')
plt.show()""")

md("""## 3. 博弈论视角 🪙

### 3.1 纳什均衡

```
纳什均衡 (Nash Equilibrium):
  没有智能体能通过单方面改变策略来获益

  在均衡点:
  - 每个智能体的策略是对手的最佳响应
  - 没有人有动力偏离

AlphaZero 的收敛:
  自我对弈 → 纳什均衡
  → 最优对最优
```

### 3.2 代表系统

```
OpenAI Five (Dota 2):
  - 5v5 多智能体
  - PPO + 自我对弈
  - 击败人类冠军

AlphaStar (星际争霸):
  - 多智能体联盟
  - 纳什均衡求解
  - 接近人类职业水平

Libratus/Pluribus (扑克):
  - 不完全信息博弈
  - CFR (反事实后悔最小化)
  - 击败职业选手
```

> 💡 博弈论是理解多智能体 RL 的数学基础——纳什均衡是竞争的终点。""")

code("""# 博弈论: 纳什均衡
# 囚徒困境
payoff_matrix = np.array([
    [[-1, -1], [-3, 0]],   # 合作
    [[0, -3], [-2, -2]],   # 背叛
])

print("囚徒困境:")
print("        合作    背叛")
print(f"合作  {payoff_matrix[0, 0]}  {payoff_matrix[0, 1]}")
print(f"背叛  {payoff_matrix[1, 0]}  {payoff_matrix[1, 1]}")
print("\\n纳什均衡: (背叛, 背叛) → (-2, -2)")
print("虽然 (合作, 合作) → (-1, -1) 更好")
print("但单方面偏离不划算 → 纳什均衡不一定是全局最优")

# 可视化
fig, ax = plt.subplots(figsize=(8, 6))
strategies = ['合作', '背叛']
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'({payoff_matrix[i,j,0]}, {payoff_matrix[i,j,1]})',
               ha='center', va='center', fontsize=14, fontweight='bold')
ax.scatter(1, 1, s=500, c='red', marker='*', zorder=5, label='纳什均衡')
ax.scatter(0, 0, s=300, c='green', marker='o', zorder=5, label='全局最优')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(strategies); ax.set_yticklabels(strategies)
ax.set_xlabel('玩家2', fontsize=12); ax.set_ylabel('玩家1', fontsize=12)
ax.set_title('囚徒困境: 纳什均衡 vs 全局最优', fontsize=14, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.savefig('notebooks/fig_game_theory.png', bbox_inches='tight')
plt.show()""")

md("""## 4. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| AlphaGo/AlphaZero (自我对弈+MCTS) | ✅ |
| 多智能体 RL (协作/竞争) | ✅ |
| 博弈论 (纳什均衡) | ✅ |
| 代表系统 (OpenAI Five/AlphaStar) | ✅ |

### 核心 takeaway
> **AlphaZero 用自我对弈达到超人类水平，多智能体 RL 用博弈论理解竞争**——从围棋到星际，深度 RL 不断突破。纳什均衡是竞争的数学终点。

### 🔗 下一章
**`73_gnn.ipynb`** — 图神经网络、Graph Transformer

---

> 💬 **拓展章进行中 (1/2)。**""")

output_path = "notebooks/72_deep_rl_frontiers.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")