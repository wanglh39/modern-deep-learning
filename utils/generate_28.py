# 生成 28_embodied_simulation.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 28 — 具身仿真环境：Isaac Gym / Habitat

> 仿真器是具身智能的"健身房"——在虚拟世界训练，迁移到真实机器人。
> Isaac Gym 做高频物理，Habitat 做导航，MuJoCo 做控制。

## 本章你将掌握

1. **仿真器的作用**：为什么需要仿真
2. **主流仿真器对比**：Isaac Gym / Habitat / MuJoCo / PyBullet
3. **并行环境**：向量化加速训练
4. **仿真到现实的桥梁**""")

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

md("""## 1. 为什么需要仿真器

### 1.1 真实机器人的困难

```
真实机器人训练:
  - 速度: 每秒几个 step (机械臂响应慢)
  - 成本: 机器人昂贵, 易损坏
  - 安全: 训练中可能伤人/伤物
  - 数据: 收集慢, 17个月才13万 episode (RT-1)
  - 重置: 每次实验后要手动复位
```

### 1.2 仿真器的优势

```
仿真训练:
  - 速度: 每秒几千~几万 step (GPU 并行)
  - 成本: 一次购买, 无限使用
  - 安全: 虚拟碰撞无损失
  - 数据: 自动生成, 无限量
  - 重置: 一键复位
```

### 1.3 仿真器的核心组件

```
物理引擎:  模拟重力、碰撞、摩擦、关节
渲染器:    生成摄像头图像 (RGB/深度)
场景:      3D 环境 (房间/地形/物体)
机器人:    URDF/MJCF 描述的机器人模型
传感器:    模拟摄像头、IMU、触觉
```

> 💡 仿真器让 RL 训练成为可能——在仿真中跑百万次试验，真实世界做不到。""")

md("""## 2. 主流仿真器对比

### 2.1 四大仿真器

| 仿真器 | 物理引擎 | 特点 | 典型用途 |
|--------|---------|------|---------|
| **Isaac Gym** | PhysX | GPU 并行, 速度快 | 机械臂操作 |
| **Isaac Lab** | PhysX | Isaac Gym 继任 | 通用 |
| **Habitat** | Bullet | 导向导航 | 视觉导航 |
| **MuJoCo** | MuJoCo | 精确, 接触好 | 控制/RL |
| **PyBullet** | Bullet | Python 友好 | 研究/教学 |

### 2.2 选择指南

```
要训练机械臂操作 → Isaac Gym/Lab (GPU 并行)
要训练视觉导航   → Habitat (室内场景)
要研究控制理论   → MuJoCo (精确物理)
要快速原型       → PyBullet (简单)
```

### 2.3 Isaac Gym 的突破

Isaac Gym 把物理仿真放到 GPU 上：
- 4096 个环境并行
- 每秒 10 万+ step
- 训练时间从天级降到分钟级

> 💡 Isaac Gym 的 GPU 并行是革命性的——RL 训练从"天"到"小时"。
> 但它已停止维护，继任者是 Isaac Lab（基于 Isaac Sim）。""")

code("""# 仿真器并行环境模拟
class ParallelEnvs:
    def __init__(self, n_envs=4096):
        self.n_envs = n_envs
        # 模拟: 每个环境的状态
        self.positions = torch.randn(n_envs, 3)
        self.velocities = torch.zeros(n_envs, 3)
        self.steps = 0

    def step(self, actions):
        # 批量物理 step (所有环境同时)
        self.velocities += actions * 0.01
        self.positions += self.velocities * 0.01
        self.positions = torch.clamp(self.positions, -5, 5)
        self.steps += 1

        # 批量计算奖励 (到达目标)
        target = torch.tensor([1.0, 1.0, 1.0])
        distances = (self.positions - target).norm(dim=-1)
        rewards = -distances
        dones = distances < 0.1
        return self.positions, rewards, dones

    def reset(self, done_mask):
        # 批量重置已完成的环境
        n_reset = done_mask.sum().item()
        if n_reset > 0:
            self.positions[done_mask] = torch.randn(n_reset, 3) * 0.1
            self.velocities[done_mask] = 0

# 演示并行环境
envs = ParallelEnvs(n_envs=4096)
import time

start = time.time()
for _ in range(1000):
    actions = torch.randn(4096, 3) * 0.1
    obs, rewards, dones = envs.step(actions)
    envs.reset(dones)
elapsed = time.time() - start

steps_per_sec = 1000 * 4096 / elapsed
print(f"4096 个并行环境, 1000 步")
print(f"耗时: {elapsed:.3f}s")
print(f"总 step 数: {1000 * 4096}")
print(f"每秒 step: {steps_per_sec:.0f}")
print(f"完成的环境: {dones.sum().item()}")
print("Isaac Gym: GPU 并行让 RL 训练速度提升几个数量级。")""")

md("""## 3. 向量化环境：RL 训练的标准范式

### 3.1 为什么向量化

```
单环境 RL:
  for each step:
    action = policy(obs)
    obs, reward = env.step(action)
  → 每次只处理一个环境, GPU 利用率低

向量化 RL:
  for each step:
    actions = policy(all_obs)      # 批量
    all_obs, all_rewards = envs.step(actions)  # 批量
  → N 个环境同时, GPU 满载
```

### 3.2 PPO + 向量化环境

```python
# 典型训练循环
envs = ParallelEnvs(n=4096)
for update in range(1000):
    # 收集 4096 个环境的经验
    for _ in range(rollout_len):
        actions = policy(obs)
        obs, rewards, dones = envs.step(actions)

    # PPO 更新
    loss = ppo_loss(rollout)
    loss.backward()
    optimizer.step()
```

> 💡 向量化环境是 RL 训练的标配——没有它，RL 慢得不可行。
# Stable Baselines3, CleanRL 等库都支持。""")

code("""# 向量化 PPO 训练模拟
def vectorized_ppo_demo(n_envs=256, n_updates=50):
    # 简单策略网络
    policy = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 3))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # 256 个并行环境
    envs = ParallelEnvs(n_envs=n_envs)

    rewards_history = []
    for update in range(n_updates):
        # 收集经验
        batch_rewards = 0
        for _ in range(10):  # rollout length
            obs = envs.positions
            actions = policy(obs) + torch.randn(n_envs, 3) * 0.1
            obs, rewards, dones = envs.step(actions)
            envs.reset(dones)
            batch_rewards += rewards.mean().item()

        # 简化 PPO 更新 (只做策略梯度)
        loss = -batch_rewards  # 简化
        optimizer.zero_grad()
        # 模拟梯度回传
        for p in policy.parameters():
            p.grad = torch.randn_like(p) * 0.01
        optimizer.step()

        rewards_history.append(batch_rewards / 10)

    return rewards_history

rewards = vectorized_ppo_demo()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rewards, 'b-', linewidth=2)
ax.set_xlabel('PPO Update'); ax.set_ylabel('平均奖励')
ax.set_title('向量化 PPO 训练 (256 环境)'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_vectorized_ppo.png', bbox_inches='tight')
plt.show()
print(f"256 环境并行 PPO: 奖励 {rewards[0]:.2f} → {rewards[-1]:.2f}")
print("向量化环境: RL 训练的标配。")""")

md("""## 4. Habitat：视觉导航仿真

### 4.1 特点

Habitat 专注于**室内视觉导航**：

```
环境: 真实 3D 扫描的室内场景 (Gibson, HM3D)
任务: PointNav (导航到目标点), ObjectNav (找到物体)
传感器: RGB/深度摄像头
动作: 前进/转向
```

### 4.2 为什么 Habitat 重要

- **真实场景**：用真实 3D 扫描，不是合成
- **视觉挑战**：光照、纹理、遮挡
- **导航基准**：导航任务的标准化评估

### 4.3 典型任务

```
PointNav:
  给定: 目标坐标 (相对位置)
  输入: RGB/深度 + 方向
  输出: 前进/转向
  目标: 到达目标点

ObjectNav:
  给定: "找到椅子"
  输入: RGB
  输出: 前进/转向
  目标: 找到并靠近椅子
```

> 💡 Habitat 推动了视觉导航的研究——从经典 SLAM 到学习导航。""")

code("""# Habitat 导航模拟
class HabitatNavEnv:
    def __init__(self, map_size=20):
        self.map_size = map_size
        self.reset()

    def reset(self):
        # 随机起点和目标
        self.position = np.array([np.random.uniform(0, self.map_size),
                                   np.random.uniform(0, self.map_size)])
        self.target = np.array([np.random.uniform(0, self.map_size),
                                np.random.uniform(0, self.map_size)])
        self.heading = np.random.uniform(0, 2 * np.pi)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        # 观察: 到目标的距离 + 方向
        delta = self.target - self.position
        distance = np.linalg.norm(delta)
        angle = np.arctan2(delta[1], delta[0]) - self.heading
        return {'distance': distance, 'angle': angle}

    def step(self, action):
        if action['type'] == 'move':
            self.position += np.array([np.cos(self.heading), np.sin(self.heading)]) * action['distance']
        elif action['type'] == 'turn':
            self.heading += action['angle']

        self.steps += 1
        obs = self._get_obs()
        done = obs['distance'] < 0.5
        reward = 10.0 if done else -0.01  # 到达奖励
        return obs, reward, done

# 简单导航策略: 转向目标, 然后前进
def navigate(env, max_steps=100):
    obs = env.reset()
    trajectory = [env.position.copy()]
    for _ in range(max_steps):
        if abs(obs['angle']) > 0.1:
            env.step({'type': 'turn', 'angle': obs['angle']})
        else:
            env.step({'type': 'move', 'distance': min(1.0, obs['distance'])})
        obs, reward, done = env.step({'type': 'move', 'distance': 0})  # 只获取 obs
        trajectory.append(env.position.copy())
        if done:
            break
    return trajectory, done

env = HabitatNavEnv()
traj, success = navigate(env)
traj = np.array(traj)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(traj[:, 0], traj[:, 1], 'b-o', linewidth=2, markersize=3, label='路径')
ax.plot(env.target[0], env.target[1], 'r*', markersize=20, label='目标')
ax.plot(traj[0, 0], traj[0, 1], 'g^', markersize=15, label='起点')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title(f'Habitat 导航 ({"成功" if success else "失败"})'); ax.legend()
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_habitat_nav.png', bbox_inches='tight')
plt.show()
print(f"导航 {'成功' if success else '失败'}, 步数: {len(traj)}")
print("Habitat: 真实 3D 场景中的视觉导航训练。")""")

md("""## 5. 仿真器的未来

### 5.1 趋势

```
1. GPU 物理加速: Isaac Gym → Isaac Lab
2. 真实感渲染: 光线追踪, 逼真图像
3. 大规模场景: 城市/地形级
4. 可微仿真: 梯度通过仿真器回传
5. 生成式仿真: 用生成模型造场景
```

### 5.2 可微仿真

传统仿真器不可微，无法端到端训练。可微仿真器（如 Brax, Genesis）允许：

```
损失 → 策略 → 动作 → 仿真 → 观察 → 损失
         ↑__________________________|
              梯度可以回传
```

### 5.3 生成式仿真

用生成模型（如视频扩散）生成训练场景：

```
文本: "客厅里有沙发和茶几"
→ 3D 场景生成
→ 在生成的场景中训练
→ 无限多样化环境
```

> 💡 仿真器是具身智能的基础设施。
> 未来：GPU 加速 + 真实渲染 + 生成式场景 → 更强的 sim-to-real。""")

code("""# 仿真器演进时间线
fig, ax = plt.subplots(figsize=(13, 5))

simulators = [
    (2012, 'MuJoCo', '精确物理'),
    (2016, 'PyBullet', 'Python 友好'),
    (2017, 'Gym', 'RL 标准接口'),
    (2019, 'Habitat', '视觉导航'),
    (2021, 'Isaac Gym', 'GPU 并行'),
    (2023, 'Isaac Lab', 'Isaac 继任'),
    (2024, 'Genesis', '可微仿真'),
]

for year, name, desc in simulators:
    ax.scatter(year, 0, s=200, zorder=5, color='steelblue')
    ax.annotate(f'{name}\\n({desc})', xy=(year, 0), xytext=(year, 0.3 + 0.2 * (year % 2)),
                ha='center', fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlim(2010, 2026); ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('年份'); ax.set_title('具身仿真器演进')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_yticks([])
plt.tight_layout()
plt.savefig('notebooks/fig_sim_evolution.png', bbox_inches='tight')
plt.show()
print("仿真器: 从 CPU 到 GPU, 从固定到可微, 从手工到生成式。")""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 仿真器的作用 | ✅ |
| 主流仿真器对比 | ✅ |
| 向量化环境 | ✅ |
| Habitat 导航 | ✅ |
| 仿真器趋势 | ✅ |

### 核心 takeaway
> **仿真器是具身智能的"健身房"**。
> Isaac Gym 的 GPU 并行让 RL 训练从天级到分钟级。
> 未来方向：可微仿真 + 生成式场景 + 真实渲染。

### 🔗 下一章
**`29_speech_audio.ipynb`** — ASR、TTS、Whisper

---

> 💬 **板块四进行中。**""")

output_path = "notebooks/28_embodied_simulation.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")