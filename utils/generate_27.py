# 生成 27_embodied_ai.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 27 — 具身智能：VLA、RT-2 与机器人

> 🧭 让 AI 不只看图说话，而是**看图动手**。
> VLA（Vision-Language-Action）模型把 LLM 的推理能力接到机器人上。

## 本章你将掌握

1. **具身智能的挑战**：感知→规划→执行
2. **RT-1/RT-2**：Google 的机器人 Transformer
3. **VLA 架构**：视觉+语言→动作
4. **模仿学习 vs RL**：机器人学习范式""")

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

md("""## 1. 具身智能的挑战

### 1.1 什么是具身智能

```
传统 AI:  输入 → 输出 (纯数字)
具身 AI:  感知环境 → 规划 → 执行动作 → 观察结果 → 循环

例子: 机器人收拾桌子
  观察: 看到桌上有杯子、书、笔
  规划: 先拿杯子放架子, 再整理书, 最后收笔
  执行: 控制机械臂抓取
  观察: 看到桌子变干净了
```

### 1.2 三大挑战

| 挑战 | 说明 |
|------|------|
| **感知** | 从摄像头/触觉理解场景 |
| **规划** | 把高层目标分解为动作序列 |
| **执行** | 精确控制机械臂（毫米级） |
| **泛化** | 新物体、新场景、新指令 |

### 1.3 为什么 LLM 有用

```
传统机器人: 硬编码规则 or 从 demonstrations 学简单策略
VLA:       LLM 提供常识推理 + 语言理解 → 动作

"把蓝色杯子放到左边" → LLM 理解"蓝色""左边" → 输出动作
```

> 💡 LLM 的常识推理是具身智能的"大脑"，机械臂是"身体"。
> VLA 把两者连接起来。""")

md("""## 2. RT-1：Robotics Transformer

### 2.1 架构

RT-1（Robotics Transformer 1）是 Google 2022 年的工作：

```
输入:
  图像历史 (最近几帧)
  语言指令 ("拿杯子")

输出:
  动作 token: [夹爪, 位置x, 位置y, 位置z, 旋转, ...]

架构: ViT (图像) + Tokenizer (语言) + Transformer (动作)
```

### 2.2 动作离散化

RT-1 把连续动作**离散化为 token**：

```
夹爪: {开, 关} → 2 个 token
位置 x: [-1, 1] → 256 个 bin (离散化)
旋转: SO(3) → 离散化

→ 每个动作 = 11 个 token
→ 机器人控制变成"翻译"任务
```

### 2.3 训练数据

RT-1 用 **13 万** 个机器人 episode 训练：
- 每天在真实机器人上收集
- 历时 17 个月
- 780 个不同任务

> 💡 RT-1 的关键：把机器人控制变成 token 预测，
> 这样可以直接用 Transformer 架构。""")

code("""# RT-1 简化实现
class MiniRT1(nn.Module):
    def __init__(self, d_model=256, n_action_tokens=11, action_vocab=256):
        super().__init__()
        # 图像编码器 (简化)
        self.image_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 3, d_model),
            nn.LayerNorm(d_model)
        )
        # 语言编码器 (简化)
        self.lang_encoder = nn.Linear(128, d_model)
        # Transformer 主体
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*4, batch_first=True),
            num_layers=4
        )
        # 动作输出头
        self.action_head = nn.Linear(d_model, action_vocab)

    def forward(self, image, lang_embed):
        # 编码图像
        img_token = self.image_encoder(image).unsqueeze(1)  # [B, 1, d]
        # 编码语言
        lang_token = self.lang_encoder(lang_embed).unsqueeze(1)  # [B, 1, d]
        # 拼接
        tokens = torch.cat([img_token, lang_token], dim=1)  # [B, 2, d]
        # Transformer
        output = self.transformer(tokens)
        # 预测动作 (简化: 只预测1个 token)
        action_logits = self.action_head(output[:, 0])  # [B, action_vocab]
        return action_logits

# 演示 RT-1
rt1 = MiniRT1()
image = torch.randn(4, 64, 64, 3)  # 4个场景
lang_embed = torch.randn(4, 128)   # "拿杯子" 的嵌入

action_logits = rt1(image, lang_embed)
action = action_logits.argmax(dim=-1)
print(f"输入: 图像 {image.shape} + 语言嵌入 {lang_embed.shape}")
print(f"输出: 动作 logits {action_logits.shape}")
print(f"预测动作 token: {action.tolist()}")
print(f"参数量: {sum(p.numel() for p in rt1.parameters())/1e6:.1f}M")
print("RT-1: 图像+语言 → 动作 token (离散化)。")""")

md("""## 3. RT-2：VLA 的突破

### 3.1 核心创新

RT-2（2023）把 **VLM**（如 PaLI-X）直接变成 **VLA**：

```
RT-1: 专门设计的机器人 Transformer
RT-2: 复用 VLM (PaLI-X, 540B 参数)

关键: 把动作变成文本 token
  动作 (x=0.5, y=0.3, gripper=close)
  → "x:0.5 y:0.3 gripper:close"
  → VLM 直接输出这段"文本"
```

### 3.2 优势

- **继承 VLM 知识**：RT-2 知道"什么是杯子""左边是哪边"
- **更好的泛化**：能处理训练时没见过的物体/指令
- **链式推理**：复杂任务可以分步规划

### 3.3 结果

```
RT-1: 只会训练时见过的任务
RT-2: 能执行新指令, 如"把草莓放进碗" (训练时没见过)
      能理解语义, 如"把东西移到有太阳标志的一边"
```

> 💡 RT-2 的哲学：**机器人控制 = 特殊格式的文本生成**。
> 这样可以复用 VLM 的全部能力——常识、推理、泛化。""")

code("""# RT-2: 动作作为文本
def action_to_text(action):
    # 把连续动作转成文本
    x, y, z, gripper = action
    gripper_str = "close" if gripper > 0.5 else "open"
    return f"x:{x:.2f} y:{y:.2f} z:{z:.2f} gripper:{gripper_str}"

def text_to_action(text):
    # 解析文本动作
    parts = text.split()
    action = {}
    for part in parts:
        key, val = part.split(':')
        if key == 'gripper':
            action[key] = 1.0 if val == 'close' else 0.0
        else:
            action[key] = float(val)
    return action

# 演示
actions = [
    (0.5, 0.3, 0.2, 1.0),  # 抓取
    (0.8, 0.6, 0.4, 0.0),  # 放下
    (0.2, 0.1, 0.3, 1.0),  # 移动
]

print("RT-2: 动作 ↔ 文本")
for a in actions:
    text = action_to_text(a)
    parsed = text_to_action(text)
    print(f"  动作 {a} → 文本 '{text}' → 解析 {parsed}")

print("\\nVLM 输出文本 → 解析为机器人动作——无需专门的动作头!")""")

md("""## 4. VLA 架构对比

### 4.1 三种 VLA 范式

```
1. RT-1 式: 专门设计 (图像+语言 → 动作 token)
   优点: 高效; 缺点: 不继承 VLM 知识

2. RT-2 式: VLM + 动作文本化
   优点: 继承 VLM 知识; 缺点: 大模型, 慢

3. OpenVLA 式: 开源 VLA (基于 Llama)
   优点: 开源可复现; 缺点: 效果略差
```

### 4.2 代表模型

| 模型 | 基础 | 特点 |
|------|------|------|
| **RT-1** | EfficientNet | 专用架构 |
| **RT-2** | PaLI-X (540B) | VLM → VLA |
| **OpenVLA** | Llama 2 | 开源 |
| **Octo** | ViT | 通用策略 |
| **π₀** | PaLI | Flow Matching 动作 |

> 💡 2024 年趋势：开源 VLA（OpenVLA, Octo）让更多研究者能参与。""")

code("""# VLA 架构对比可视化
fig, ax = plt.subplots(figsize=(12, 6))

models = ['RT-1\\n(2022)', 'RT-2\\n(2023)', 'OpenVLA\\n(2024)', 'Octo\\n(2024)', 'π₀\\n(2024)']
params = [35, 540, 7, 93, 300]  # 参数量 (M)
generalization = [40, 85, 70, 65, 80]  # 泛化能力 (%)
speed = [90, 30, 60, 70, 50]  # 推理速度 (相对)

x = np.arange(len(models))
width = 0.25

ax.bar(x - width, params, width, label='参数量(M)', color='steelblue', alpha=0.8)
ax.bar(x, generalization, width, label='泛化能力(%)', color='coral', alpha=0.8)
ax.bar(x + width, speed, width, label='速度(相对)', color='forestgreen', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('值')
ax.set_title('VLA 模型对比')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/fig_vla_compare.png', bbox_inches='tight')
plt.show()
print("RT-2 泛化最好但大; RT-1 快但泛化弱; OpenVLA 开源平衡。")""")

md("""## 5. 机器人学习范式

### 5.1 模仿学习 (BC)

```
数据: 专家演示 (人类遥操作)
  {(观察_i, 动作_i)}

训练: 行为克隆 (BC)
  学 π(a|o) 使 π 匹配专家分布

优点: 简单, 稳定
缺点: 需要大量演示, 无法超越专家
```

### 5.2 强化学习 (RL)

```
环境: 机器人模拟器
奖励: 任务完成 → +1

训练: PPO/SAC 等
  学 π(a|o) 最大化 E[Σ γ^t r_t]

优点: 能超越人类, 发现新策略
缺点: 奖励设计难, 模拟到现实差距
```

### 5.3 从人类反馈学习 (RLHF for Robotics)

```
1. BC 预训练 (从演示)
2. RL 微调 (从奖励)
3. RLHF (从人类偏好)

→ 结合 BC 的稳定性和 RL 的探索能力
```

> 💡 当前主流：**BC 预训练 + RL 微调**。
> 纯 RL 在真实机器人上太难训练，纯 BC 无法超越人类。""")

code("""# 模仿学习 vs RL 对比
def behavior_cloning_demo(n_expert=200, n_epochs=100):
    # 模拟专家策略
    torch.manual_seed(42)
    expert_w = torch.randn(4, 1)

    # 生成专家数据
    observations = torch.randn(n_expert, 4)
    expert_actions = observations @ expert_w + 0.1 * torch.randn(n_expert, 1)

    # BC: 学一个策略网络
    policy = nn.Linear(4, 1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    losses = []
    for _ in range(n_epochs):
        pred = policy(observations)
        loss = F.mse_loss(pred, expert_actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

def rl_demo(n_epochs=100):
    # 模拟 RL: 奖励驱动探索
    torch.manual_seed(42)
    rewards = []
    cumulative = 0
    for i in range(n_epochs):
        # RL 探索: 初期低, 逐渐提升
        r = 0.8 * (1 - np.exp(-i / 20)) + 0.05 * np.random.randn()
        cumulative = 0.99 * cumulative + r
        rewards.append(cumulative)
    return rewards

bc_losses = behavior_cloning_demo()
rl_rewards = rl_demo()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(bc_losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('BC Loss')
ax.set_title('模仿学习 (行为克隆)'); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(rl_rewards, 'r-', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('累积奖励')
ax.set_title('强化学习'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_robot_learning.png', bbox_inches='tight')
plt.show()
print("BC: 快速收敛到专家水平; RL: 探索后可能超越专家。")""")

md("""## 6. Sim-to-Real：从仿真到现实

### 6.1 问题

在仿真器训练的策略，放到真实机器人上往往失败：

```
仿真: 物理精确, 但不真实 (摩擦/质量/传感器有误差)
现实: 真实, 但训练慢且危险

差距: 仿真策略 → 现实 → 性能下降
```

### 6.2 解决方案

| 方法 | 思想 |
|------|------|
| **Domain Randomization** | 随机化仿真参数，增强鲁棒性 |
| **Domain Adaptation** | 学一个仿真→现实的映射 |
| **Real2Sim** | 用真实数据校准仿真 |
| **Co-training** | 仿真+真实数据联合训练 |

> 💡 Domain Randomization 最实用：在仿真中随机化摩擦、质量、光照等，
> 让策略对分布偏移鲁棒。RT-1/RT-2 都用了这个技巧。""")

code("""# Domain Randomization 演示
def domain_randomization_demo():
    # 模拟: 在不同参数下训练
    n_envs = 20
    np.random.seed(42)

    # 不随机化: 固定参数
    fixed_perf = []
    for _ in range(50):
        # 真实参数与训练参数有偏差
        real_friction = 0.5 + 0.1 * np.random.randn()
        train_friction = 0.5  # 固定
        gap = abs(real_friction - train_friction)
        perf = max(0, 1 - gap * 3)
        fixed_perf.append(perf)

    # 随机化: 训练时参数随机
    rand_perf = []
    for _ in range(50):
        real_friction = 0.5 + 0.1 * np.random.randn()
        train_friction = 0.5 + 0.2 * np.random.randn()  # 随机化
        gap = abs(real_friction - train_friction)
        perf = max(0, 1 - gap * 1.5)  # 更鲁棒
        rand_perf.append(perf)

    return fixed_perf, rand_perf

fixed, rand = domain_randomization_demo()
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(fixed, bins=15, alpha=0.7, label='固定参数', color='red', density=True)
ax.hist(rand, bins=15, alpha=0.7, label='Domain Randomization', color='green', density=True)
ax.set_xlabel('真实环境性能'); ax.set_ylabel('密度')
ax.set_title('Domain Randomization: Sim-to-Real'); ax.legend()
plt.tight_layout()
plt.savefig('notebooks/fig_sim2real.png', bbox_inches='tight')
plt.show()
print(f"固定参数: 平均性能 {np.mean(fixed):.2f}")
print(f"Domain Randomization: 平均性能 {np.mean(rand):.2f}")
print("随机化训练参数 → 策略对现实偏差更鲁棒。")""")

md("""## 7. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 具身智能挑战 | ✅ |
| RT-1 动作离散化 | ✅ |
| RT-2 VLM→VLA | ✅ |
| VLA 架构对比 | ✅ |
| BC vs RL | ✅ |
| Sim-to-Real | ✅ |

### 核心 takeaway
> **VLA = VLM + 动作**。RT-2 把机器人控制变成文本生成，继承 LLM 的常识和推理。
> 从 RT-1 到 RT-2，机器人从"专用"走向"通用"。
> Sim-to-Real 是具身智能的关键工程挑战。

### 🔗 下一章
**`28_embodied_simulation.ipynb`** — Isaac Gym/Habitat 仿真环境

---

> 💬 **板块四进行中。**""")

output_path = "notebooks/27_embodied_ai.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")