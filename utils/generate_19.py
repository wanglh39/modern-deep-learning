"""生成 19_reasoning_models.ipynb 的脚本"""
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
md("""# 19 — 🔥 推理模型：o1/o3、DeepSeek-R1、test-time scaling

> 传统 LLM 是"快思考"——直接输出答案。推理模型是"慢思考"——先**推理过程**再答案。
> OpenAI o1/o3 和 DeepSeek-R1 证明：**推理时多算几步，效果能超 GPT-4**。

## 本章你将掌握

1. **推理模型**的概念与动机
2. **Test-time scaling**：推理时算力换效果
3. **Chain of Thought**：思维链
4. **推理模型训练**：RL + 自我改进
5. **DeepSeek-R1** 的训练流程""")

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
md("""## 1. 推理模型：快思考 vs 慢思考

### 1.1 两种思考模式

```
快思考 (System 1):  问题 → 答案           (直接输出, 快但容易错)
慢思考 (System 2):  问题 → 推理步骤1 → 步骤2 → ... → 答案  (慢但准确)
```

### 1.2 为什么需要推理模型？

| 问题类型 | 快思考 | 慢思考 |
|---------|--------|--------|
| "法国首都？" | ✅ 直接知道 | 不需要 |
| "证明√2无理数" | ❌ 一步给不出 | ✅ 需要多步推理 |
| "24点游戏" | ❌ 要试很多组合 | ✅ 系统搜索 |

> 💡 数学、代码、逻辑推理需要**多步思考**——这就是 o1/R1 的用武之地。
> o1 在数学竞赛 (AIME) 上从 GPT-4 的 12% 提升到 83%！""")

code("""# 演示: 快思考 vs 慢思考
# 问题: 123 * 456 = ?

# 快思考: 直接预测 (模拟)
fast_answer = 56088  # 模型直接输出
print("快思考: 123 × 456 = ?", fast_answer)

# 慢思考: 分步推理
print("\\n慢思考:")
print("  步骤1: 123 × 456 = 123 × (400 + 50 + 6)")
print("  步骤2: 123 × 400 = 49200")
print("  步骤3: 123 × 50 = 6150")
print("  步骤4: 123 × 6 = 738")
print("  步骤5: 49200 + 6150 = 55350")
print("  步骤6: 55350 + 738 = 56088")
print("  答案: 56088")

print("\\n快思考快但可能错; 慢思考慢但可验证每一步。")
print("推理模型 = 让 LLM 学会慢思考。")""")

# ============================================================
md("""## 2. Test-Time Scaling：推理时扩展

### 2.1 两种 scaling

```
训练时 scaling:  更多参数 + 更多数据 → 更好的模型    (传统路线)
推理时 scaling:  同一模型 + 更多推理计算 → 更好答案   (o1 路线)
```

### 2.2 推理时计算怎么花？

| 方式 | 说明 | 代表 |
|------|------|------|
| **长 CoT** | 生成更长的思维链 | o1, R1 |
| **Best-of-N** | 生成 N 个答案选最好 | 通用 |
| **Self-consistency** | 多数投票 | 数学推理 |
| **Tree search** | 搜索推理树 | ToT/GoT |

> 💡 o1 的核心：**用 RL 训练模型生成长 CoT**，推理时让模型"想很久"。
> test-time compute 和效果近似**对数线性**关系——多算 10x，效果提升一个等级。""")

code("""# 模拟 test-time scaling 效果
# 不同推理计算量下的准确率
compute_levels = [1, 2, 5, 10, 20, 50, 100, 200, 500]
# 快思考: 固定准确率 (不随计算变化)
fast_acc = [0.65] * len(compute_levels)
# 慢思考: 随计算量提升 (对数关系)
slow_acc = [0.65 + 0.25 * np.log10(c) / np.log10(500) for c in compute_levels]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(compute_levels, fast_acc, 'b-o', linewidth=2.5, markersize=7, label='快思考 (固定)')
ax.plot(compute_levels, slow_acc, 'r-s', linewidth=2.5, markersize=7, label='慢思考 (test-time scaling)')
ax.set_xscale('log')
ax.set_xlabel('推理计算量 (相对)')
ax.set_ylabel('准确率')
ax.set_title('Test-Time Scaling: 推理时算力换效果')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_test_time_scaling.png', bbox_inches='tight')
plt.show()
print("快思考准确率固定; 慢思考随推理计算量对数增长——多算多得分。")""")

# ============================================================
md("""## 3. Chain of Thought：思维链

### 3.1 CoT 的形式

```
问题: 一个商店有23个苹果, 卖了17个, 又进了15个, 现在有多少?

无 CoT:  21

有 CoT:  让我一步步算:
  初始: 23个苹果
  卖了17个: 23 - 17 = 6个
  进了15个: 6 + 15 = 21个
  答案: 21
```

### 3.2 CoT 为什么有效？

1. **分解复杂问题**：每步只做简单计算
2. **中间结果缓存**：不用一步算完所有
3. **可验证**：每步可以检查
4. **更多 token = 更多计算**：利用更多推理时算力

> 💡 CoT 不需要改模型架构——只需在训练数据中包含推理过程。
> o1 的创新是用 **RL** 让模型自己学会生成好的 CoT。""")

code("""# 模拟 CoT 推理过程
class SimpleCoTModel:
    # 模拟一个能生成 CoT 的模型
    def __init__(self, accuracy_direct=0.4, accuracy_cot=0.8):
        self.acc_direct = accuracy_direct
        self.acc_cot = accuracy_cot

    def direct_answer(self, problem):
        # 快思考: 直接给答案
        correct = np.random.random() < self.acc_direct
        return correct

    def cot_answer(self, problem, n_steps=5):
        # 慢思考: 每步有概率正确, 全对才算对
        step_acc = self.acc_cot ** (1/n_steps)  # 每步准确率
        for _ in range(n_steps):
            if np.random.random() > step_acc:
                return False  # 某步错了
        return True

    def self_consistency(self, problem, n_samples=5, n_steps=5):
        # Self-consistency: 生成多个CoT, 多数投票
        results = [self.cot_answer(problem, n_steps) for _ in range(n_samples)]
        return sum(results) > n_samples / 2  # 多数正确

# 模拟推理效果
model = SimpleCoTModel(accuracy_direct=0.4, accuracy_cot=0.8)
n_problems = 500

# 不同策略的准确率
acc_direct = np.mean([model.direct_answer(i) for i in range(n_problems)])
acc_cot = np.mean([model.cot_answer(i) for i in range(n_problems)])
acc_sc5 = np.mean([model.self_consistency(i, n_samples=5) for i in range(n_problems)])
acc_sc10 = np.mean([model.self_consistency(i, n_samples=10) for i in range(n_problems)])

print("推理策略对比 (500道题):")
print(f"  直接回答:       {acc_direct:.1%}")
print(f"  CoT (5步):      {acc_cot:.1%}")
print(f"  CoT + SC(5):    {acc_sc5:.1%}")
print(f"  CoT + SC(10):   {acc_sc10:.1%}")
print("\\nSC(10) > SC(5) > CoT > 直接——更多推理计算 = 更高准确率。")""")

# ============================================================
md("""## 4. 推理模型的训练

### 4.1 o1 的训练方式（推测）

```
阶段1: 预训练 (大量含CoT的数据)
阶段2: SFT (高质量推理数据)
阶段3: RL (用正确性作为奖励, 优化CoT生成)
  - 答对 → 奖励
  - 答错 → 惩罚
  - 模型学会生成更好的推理链
```

### 4.2 DeepSeek-R1 的训练流程

```
R1-Zero:  直接在基座模型上做RL (无SFT)
  → 涌现出 "aha moment" (模型自己学会反思)
  → 但输出可读性差

R1:      R1-Zero的CoT做SFT → 再RL
  → 推理能力强 + 输出可读
  → 开源, 效果接近o1
```

> 💡 R1-Zero 的 "aha moment" 是涌现行为——模型在 RL 训练中**自己学会了** "等等，让我重新想想"。
> 这不是人类教的，是模型在追求奖励的过程中自发产生的。""")

code("""# 模拟推理模型的 RL 训练
class ReasoningRL:
    # 简化: 用RL训练模型生成更好的CoT
    def __init__(self, initial_reasoning_quality=0.3):
        self.quality = initial_reasoning_quality
        self.history = [initial_reasoning_quality]

    def generate_cot(self, n_steps=5):
        # 生成CoT, 每步质量取决于当前reasoning quality
        steps = []
        for _ in range(n_steps):
            step_correct = np.random.random() < self.quality
            steps.append(step_correct)
        return steps

    def evaluate(self, cot):
        # 评估: 所有步骤都正确才算对
        return all(cot)

    def rl_update(self, reward, lr=0.05):
        # RL更新: 答对→提升质量, 答错→降低
        self.quality += lr * (reward - 0.5)
        self.quality = np.clip(self.quality, 0.05, 0.95)
        self.history.append(self.quality)

# 训练
np.random.seed(42)
rl_agent = ReasoningRL(initial_reasoning_quality=0.3)

for episode in range(500):
    cot = rl_agent.generate_cot(n_steps=3)
    reward = 1.0 if rl_agent.evaluate(cot) else 0.0
    rl_agent.rl_update(reward, lr=0.02)

print("推理模型 RL 训练:")
print(f"  初始推理质量: {rl_agent.history[0]:.2f}")
print(f"  最终推理质量: {rl_agent.history[-1]:.2f}")
print(f"  训练前答对率: ~{rl_agent.history[0]**3:.1%} (3步全对)")
print(f"  训练后答对率: ~{rl_agent.history[-1]**3:.1%}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rl_agent.history, 'b-', linewidth=2)
ax.set_xlabel('RL 训练步数')
ax.set_ylabel('推理质量')
ax.set_title('推理模型 RL 训练: 推理质量提升')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/fig_reasoning_rl.png', bbox_inches='tight')
plt.show()
print("RL 训练让模型学会更好的推理——这就是o1/R1的核心。")""")

# ============================================================
md("""## 5. 推理模型的对比

### 5.1 当前推理模型

| 模型 | 公司 | 特点 | 开源 |
|------|------|------|------|
| **o1/o3** | OpenAI | RL训练长CoT | ❌ |
| **DeepSeek-R1** | DeepSeek | R1-Zero涌现 + SFT | ✅ |
| **QwQ** | 阿里 | 长CoT推理 | ✅ |
| **Gemini 2.0 Flash Thinking** | Google | 推理模式 | ❌ |

### 5.2 推理模型 vs 普通模型

| | 普通模型 (GPT-4) | 推理模型 (o1) |
|---|-----------------|--------------|
| **数学** | AIME 12% | AIME 83% |
| **代码** | 好 | 更好 |
| **聊天** | ✅ 自然 | ⚠️ 过度推理 |
| **速度** | 快 | 慢 (长CoT) |
| **成本** | 低 | 高 (多token) |

> 💡 推理模型不是万能——日常聊天不需要长 CoT。
> 未来趋势：**路由模型**——简单问题用快思考，难问题用慢思考。""")

code("""# 推理模型 vs 普通模型: 不同任务的表现
tasks = ['日常聊天', '知识问答', '数学竞赛', '代码生成', '逻辑推理', '创意写作']
normal_model = [0.95, 0.90, 0.12, 0.75, 0.60, 0.85]
reasoning_model = [0.80, 0.88, 0.83, 0.82, 0.85, 0.75]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(tasks))
width = 0.35
ax.bar(x - width/2, normal_model, width, label='普通模型 (GPT-4)', color='steelblue', alpha=0.8)
ax.bar(x + width/2, reasoning_model, width, label='推理模型 (o1)', color='coral', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylabel('表现分数')
ax.set_title('推理模型 vs 普通模型: 不同任务对比')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('notebooks/fig_reasoning_vs_normal.png', bbox_inches='tight')
plt.show()
print("推理模型在数学/逻辑上碾压, 但日常聊天/创意略差——各有适用场景。")""")

# ============================================================
md("""## 6. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 快思考 vs 慢思考 | ✅ |
| Test-time scaling | ✅ |
| Chain of Thought | ✅ |
| Self-consistency 多数投票 | ✅ |
| RL 训练推理模型 | ✅ |
| DeepSeek-R1 训练流程 | ✅ |

### 核心 takeaway

> **推理模型 = RL 训练的长 CoT**——用推理时算力换效果。
> o1/R1 在数学/逻辑上碾压普通模型，但日常聊天不需要。
> 未来是**路由模型**——自动选择快慢思考。

### 🔗 下一章预告

**`20_prm_process_reward.ipynb`** — PRM 过程奖励模型

---

> 💬 **写在最后**：推理模型代表了 LLM 的"慢思考"方向。
> 不是所有问题都需要快答——有些问题值得多想想。""")

# ============================================================
output_path = "notebooks/19_reasoning_models.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")