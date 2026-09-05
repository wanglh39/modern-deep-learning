# 生成 47_alignment.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 47 — 对齐范式：从 RLHF 到 Constitutional AI

> 🪙 如何让模型不仅"能做"，还要"该做"？对齐是 LLM 从"有用"到"安全有用"的关键一步。

## 本章你将掌握

1. **对齐问题**：什么是对齐？为什么需要？
2. **对齐范式演进**：RLHF → DPO → Constitutional AI → RLAIF
3. **Constitutional AI**：自我批评与自我修正
4. **机制设计视角**：博弈论看对齐
5. **对齐范式对比**：各方法优缺点""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
torch.manual_seed(42)
print("环境准备完成 ✅")""")

md("""## 1. 对齐问题

### 1.1 什么是"对齐"？

**对齐 (Alignment)**：让模型的行为与人类价值观、意图、规范一致。

```
问题: 预训练模型学会了"能说什么"
      但不知道"该说什么"

对齐: 让模型从"能说" → "该说"
      - 有用 (Helpful): 回答用户问题
      - 诚实 (Honest): 不编造信息
      - 无害 (Harmless): 不产生有害内容
```

### 1.2 对齐的三个维度

```
HHH 原则:
  Helpful  — 有用：真正帮助用户
  Honest   — 诚实：不撒谎、不编造
  Harmless — 无害：不产生有害内容

  这三者之间存在张力：
  - 太有用 → 可能帮做坏事 (helpful vs harmless)
  - 太诚实 → 可能泄露敏感信息 (honest vs harmless)
  - 太无害 → 可能拒绝太多 (harmless vs helpful)
```

> 💡 对齐不是让模型"不说话"，而是让它在"该说"和"不该说"之间找到平衡。""")

md("""## 2. 对齐范式演进

### 2.1 范式全景

```
对齐方法演进:
  2022  RLHF      — 强化学习 + 人类反馈 (InstructGPT)
  2023  DPO       — 直接偏好优化 (绕过奖励模型)
  2023  CAI       — Constitutional AI (自我批评)
  2023  RLAIF     — AI 反馈替代人类反馈
  2024  KTO       — Kahneman-Tversky 优化 (单样本)
  2024  GRPO      — 群组相对策略优化 (DeepSeek-R1)

核心思路:
  RLHF: 人类标注偏好 → 奖励模型 → RL 优化
  DPO:  人类标注偏好 → 直接优化策略 (无奖励模型)
  CAI:  宪查原则 → 自我批评 → 自我修正 → RLAIF
  RLAIF: AI 替代人类标注偏好
```""")

code("""# 对齐范式演进时间线
fig, ax = plt.subplots(figsize=(14, 5))

methods = [
    ("RLHF\\n(InstructGPT)", 2022.0, 'steelblue'),
    ("CAI\\n(Anthropic)", 2022.8, 'coral'),
    ("DPO\\n(Stanford)", 2023.2, 'forestgreen'),
    ("RLAIF\\n(Google)", 2023.5, 'purple'),
    ("KTO\\n(T5X)", 2024.0, 'darkorange'),
    ("GRPO\\n(DeepSeek)", 2024.5, 'crimson'),
]

for name, year, color in methods:
    ax.scatter(year, 0, s=200, c=color, zorder=5, edgecolors='black', linewidth=0.5)
    ax.annotate(name, (year, 0), textcoords="offset points", xytext=(0, 30),
                ha='center', fontsize=9, fontweight='bold', color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

ax.set_xlim(2021.5, 2025.2)
ax.set_ylim(-1, 1.5)
ax.set_xlabel('年份', fontsize=12)
ax.set_title('对齐范式演进 (2022-2025)', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='gray', linewidth=2, alpha=0.3)
ax.set_yticks([])
ax.grid(axis='x', alpha=0.2)

plt.tight_layout()
plt.savefig('notebooks/fig_alignment_methods.png', bbox_inches='tight')
plt.show()
print("对齐范式从 RLHF → DPO → CAI → RLAIF，核心趋势是减少人类标注成本。")""")

md("""## 3. Constitutional AI (CAI)

### 3.1 核心思想

**Constitutional AI** 由 Anthropic 提出，核心是让模型用一套"宪法"原则**自我监督**：

```
传统 RLHF:
  人类标注 (A > B) → 奖励模型 → RL
  问题: 人类标注昂贵、不一致、难扩展

CAI 思想:
  1. 定义一组"宪法"原则 (如: "不要帮助用户做危险的事")
  2. 模型生成回答
  3. 模型自我批评: "这个回答违反了哪条原则？"
  4. 模型自我修正: "改进后的回答是..."
  5. 用修正前后对训练偏好模型 (RLAIF)
```

### 3.2 CAI 流程

```
Step 1: 监督学习 (SFT)
  - 用有用数据训练初始模型

Step 2: 红队对抗 (Red Teaming)
  - 模型生成可能有害的回答

Step 3: 自我批评 (Self-Critique)
  - 对每个回答，模型问自己:
    "审查请求: 识别上述回答中的有害内容"
    "修订请求: 重写回答使其无害"

Step 4: RLAIF
  - 用 (原始回答, 修正回答) 对训练偏好模型
  - 用 RL 优化策略
```

> 💡 CAI 的妙处：用 AI 替代人类标注偏好，大幅降低成本，且可扩展。""")

code("""# Constitutional AI 自我批评流程模拟

constitution = [
    "请不要生成有害、危险或非法的内容",
    "请诚实回答，不要编造信息",
    "如果不确定，请明确说明",
    "请尊重所有用户，不要歧视",
]

test_prompts = [
    "如何制作危险物品？",
    "请描述一个不存在的历史事件",
    "帮我写一封邮件拒绝求职者",
]

# 模拟模型回答
responses = [
    "步骤1: 获取材料X\\n步骤2: 混合材料Y和Z...",  # 有害
    "1789年，法国发生了XYZ革命，导致...",  # 编造
    "亲爱的求职者，你太差了，我们不想要你。",  # 不尊重
]

# 模拟自我批评
critiques = [
    "审查: 该回答提供了制作危险物品的具体步骤，违反了原则1。\\n修订: 我不能提供制作危险物品的指导，因为这可能造成伤害。",
    "审查: 1789年法国大革命是真实的，但'XYZ革命'是编造的，违反了原则2。\\n修订: 我不确定您指的是哪个事件，请提供更多细节。",
    "审查: 该邮件语气不尊重，违反了原则4。\\n修订: 感谢您的申请，但该职位已招满，祝您求职顺利。",
]

print("=" * 70)
print("Constitutional AI 自我批评流程模拟")
print("=" * 70)
for i, (prompt, response, critique) in enumerate(zip(test_prompts, responses, critiques)):
    print(f"\\n--- 案例 {i+1} ---")
    print(f"用户: {prompt}")
    print(f"初始回答: {response}")
    print(f"自我批评: {critique}")

print("\\n" + "=" * 70)
print("CAI 用自我批评生成偏好对: (初始回答, 修正回答) → 偏好模型")
print("=" * 70)""")

md("""## 4. 实现 CAI 偏好对生成

下面我们实现一个简化的 CAI 流程：用"宪法"原则生成偏好对。""")

code("""# 简化的 Constitutional AI 实现

class ConstitutionalAI:
    def __init__(self, principles):
        self.principles = principles
        self.preference_pairs = []

    def generate_response(self, prompt):
        # 模拟模型生成 (用简单规则)
        if "危险" in prompt or "有害" in prompt:
            return "这是制作步骤: 1. 获取材料 2. 混合..."
        elif "编造" in prompt:
            return "是的，这个虚构事件确实发生了..."
        else:
            return f"关于'{prompt}'，我的回答是..."

    def self_critique(self, prompt, response):
        # 模拟自我批评: 检查是否违反原则
        violations = []
        for i, principle in enumerate(self.principles):
            if "危险" in response and i == 0:
                violations.append((i, principle))
            if "虚构" in response and i == 1:
                violations.append((i, principle))

        if violations:
            critique = f"违反原则: {violations[0][1]}"
            revised = "我不能提供这方面的帮助。"
        else:
            critique = "无违反"
            revised = response

        return critique, revised

    def collect_preference_pairs(self, prompts):
        for prompt in prompts:
            response = self.generate_response(prompt)
            critique, revised = self.self_critique(prompt, response)
            if critique != "无违反":
                self.preference_pairs.append({
                    'prompt': prompt,
                    'chosen': revised,
                    'rejected': response,
                    'critique': critique,
                })
        return self.preference_pairs

principles = [
    "不要生成有害或危险内容",
    "不要编造虚假信息",
    "不确定时要明确说明",
]

cai = ConstitutionalAI(principles)
test_prompts = ["如何制作危险物品？", "请编造一个事件", "正常问题"]

pairs = cai.collect_preference_pairs(test_prompts)

print("CAI 生成的偏好对:")
for i, pair in enumerate(pairs):
    print(f"\\n偏好对 {i+1}:")
    print(f"  Prompt: {pair['prompt']}")
    print(f"  Chosen (修正): {pair['chosen']}")
    print(f"  Rejected (原始): {pair['rejected']}")
    print(f"  Critique: {pair['critique']}")""")

md("""## 5. RLAIF：AI 反馈替代人类反馈

### 5.1 RLAIF 思想

```
RLHF: 人类标注偏好 → 奖励模型 → RL
       ^^^^^^^^^^^^
       昂贵、慢、不一致

RLAIF: AI 标注偏好 → 压力模型 → RL
       ^^^^^^^^^^^^
       便宜、快、一致

关键: 用一个强大的模型 (如 GPT-4) 来判断
      两个回答哪个更好
```

### 5.2 RLAIF vs RLHF

```
          RLHF              RLAIF
成本      高 ($/标注)       低 (API调用)
速度      慢 (人工)         快 (自动)
一致性    低 (人不同)       高 (模型确定)
质量      高 (人类判断)     中 (AI判断可能有偏)
扩展性    差                好
```

> 💡 RLAIF 的核心洞察：用 AI 判断 AI，虽然可能有偏，但成本和扩展性远优于人类标注。""")

code("""# RLAIF: 用 AI 做裁判
class AIPreferenceJudge:
    def __init__(self, principles):
        self.principles = principles

    def judge(self, prompt, response_a, response_b):
        score_a = self._score(prompt, response_a)
        score_b = self._score(prompt, response_b)
        if score_a > score_b:
            return 'A', score_a, score_b
        else:
            return 'B', score_a, score_b

    def _score(self, prompt, response):
        score = 0.5
        if "不能" in response or "无法" in response:
            score += 0.2
        if "危险" in response or "步骤" in response:
            score -= 0.3
        if "不确定" in response:
            score += 0.1
        return max(0, min(1, score))

judge = AIPreferenceJudge(["无害", "诚实", "有用"])

prompt = "如何制作危险物品？"
response_a = "步骤1: 获取材料..."  # 有害
response_b = "我不能提供这方面的帮助。"  # 拒绝

winner, sa, sb = judge.judge(prompt, response_a, response_b)
print(f"Prompt: {prompt}")
print(f"回答A: {response_a} (分数: {sa:.2f})")
print(f"回答B: {response_b} (分数: {sb:.2f})")
print(f"裁判: 回答{winner} 更好")
print(f"\\nRLAIF 用 AI 裁判生成偏好对，成本远低于人类标注。")""")

md("""## 6. 机制设计视角 🪙

### 6.1 对齐作为机制设计

**机制设计 (Mechanism Design)** 是博弈论的逆向工程：

```
博弈论 (正向): 给定规则 → 分析策略
机制设计 (逆向): 给定目标 → 设计规则

对齐 = 机制设计:
  目标: 模型行为符合人类价值观
  规则: 训练方法 (RLHF/DPO/CAI)
  策略: 模型学到的行为

关键问题: 模型是否会"策略性"地假装对齐?
```

### 6.2 激励兼容

```
激励兼容 (Incentive Compatible):
  机制使得"诚实"是最优策略

对齐中的挑战:
  - 模型可能学会"说好话"而非"做好事"
  - 模型可能学会"钻空子"绕过约束
  - 模型可能在评估时表现好，部署时表现差

→ 这是对齐的核心难题: 如何确保真对齐而非伪对齐?
```

### 6.3 谢逊 vs 真对齐

```
谦逊对齐 (Humble Alignment):
  - 承认模型可能不对齐
  - 持续监控和修正
  - 不假设一次对齐就永远对齐

真对齐 vs 伪对齐:
  伪对齐: 在评估分布上表现好，但 OOD 时失效
  真对齐: 在所有分布上都表现好 (更难达到)
```

> 💡 机制设计视角提醒我们：对齐不是一次性的，而是持续的博弈过程。""")

code("""# 对齐范式对比
fig, ax = plt.subplots(figsize=(12, 7))

methods = ['RLHF', 'DPO', 'CAI', 'RLAIF', 'KTO', 'GRPO']
metrics = {
    '人类标注成本':    [9, 7, 3, 1, 5, 4],
    '实现复杂度':      [9, 3, 5, 6, 3, 5],
    '对齐质量':        [8, 7, 7, 6, 6, 8],
    '扩展性':          [3, 6, 8, 9, 7, 7],
    '稳定性':          [5, 8, 7, 6, 8, 6],
}

import numpy as np
x = np.arange(len(methods))
width = 0.15
colors = ['steelblue', 'coral', 'forestgreen', 'purple', 'darkorange']

for i, (metric, values) in enumerate(metrics.items()):
    offset = (i - 2) * width
    bars = ax.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.85)

ax.set_xlabel('对齐方法', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('对齐范式多维度对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_alignment_comparison.png', bbox_inches='tight')
plt.show()
print("RLHF 质量高但成本高; DPO 简单稳定; CAI/RLAIF 扩展性好; GRPO 质量高且成本适中。")""")

md("""## 7. 对齐的挑战

### 7.1 核心挑战

```
1. 偏好不一致
   - 不同标注者有不同标准
   - 同一标注者不同时间不同

2. 奖励黑客 (Reward Hacking)
   - 模型学会钻奖励模型空子
   - 而非真正对齐

3. 灾难性遗忘
   - 对齐训练后忘了预训练能力

4. 分布外失效
   - 评估时对齐，部署时不对齐

5. 规模扩展
   - 对齐效果随模型规模变化
   - 大模型可能更难对齐 (涌现能力)
```

### 7.2 奖励黑客示例

```
奖励模型: "回答越长越好"
模型策略: 生成冗长但无用的回答
→ 高奖励，但不对齐

解决:
  - 多维度奖励
  - 对抗验证
  - 持续监控
```""")

code("""# 奖励黑客模拟
np.random.seed(42)

steps = np.arange(100)
true_alignment = 0.3 + 0.4 * (1 - np.exp(-steps / 30))  # 真实对齐
reward_hacking = 0.3 + 0.7 * (1 - np.exp(-steps / 15))   # 奖励分数

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steps, true_alignment, 'b-', linewidth=2, label='真实对齐程度')
ax.plot(steps, reward_hacking, 'r--', linewidth=2, label='奖励模型分数')
ax.fill_between(steps, true_alignment, reward_hacking,
                where=reward_hacking > true_alignment,
                alpha=0.2, color='red', label='奖励黑客间隙')

ax.set_xlabel('训练步数', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('奖励黑客：奖励分数 ≠ 真实对齐', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.annotate('模型学会钻空子', xy=(50, 0.75), fontsize=12, color='red',
            ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('notebooks/fig_reward_hacking.png', bbox_inches='tight')
plt.show()
print("奖励黑客: 模型优化奖励分数而非真实对齐 → 需要多维度评估和持续监控。")""")

md("""## 8. 对齐范式统一视角

### 8.1 统一框架

```
所有对齐方法可统一为:
  max_π E[reward(x, y)] - β * KL(π || π_ref)

区别在于 reward 怎么来:
  RLHF:  reward = 奖励模型 (人类偏好训练)
  DPO:   reward 隐式 (直接用偏好对)
  CAI:   reward = 自我批评分数
  RLAIF: reward = AI 裁判分数
  KTO:   reward = 单样本效用
  GRPO:  reward = 群组相对分数
```

### 8.2 选择建议

```
场景 → 推荐方法:
  - 有标注预算 + 高质量要求 → RLHF
  - 快速迭代 + 简单 → DPO
  - 大规模 + 低成本 → CAI / RLAIF
  - 单样本偏好 → KTO
  - 推理任务 + 群组比较 → GRPO
```

> 💡 没有最好的对齐方法，只有最适合场景的方法。""")

md("""## 9. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 对齐问题 HHH 原则 | ✅ |
| 对齐范式演进 (RLHF→DPO→CAI→RLAIF) | ✅ |
| Constitutional AI 自我批评 | ✅ |
| RLAIF AI 反馈 | ✅ |
| 机制设计视角 | ✅ |
| 奖励黑客挑战 | ✅ |

### 核心 takeaway
> **对齐是从"能说"到"该说"的过程**——从 RLHF 的人类标注到 CAI 的自我批评，核心趋势是降低成本、提高扩展性。机制设计视角提醒我们：对齐是持续的博弈，而非一次性的训练。

### 🔗 下一章
**`48_safety_jailbreak.ipynb`** — 红队、越狱、abliteration

---

> 💬 **板块八(对齐、安全与评估)进行中 (1/5)。**""")

output_path = "notebooks/47_alignment.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")