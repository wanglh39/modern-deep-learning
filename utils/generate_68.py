# 生成 68_automated_discovery.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 68 — 自动化发现：AlphaProof 与 AlphaEvolve

> 🔥 AI 不只解决问题，还能发现新知识——数学定理、新算法。

## 本章你将掌握

1. **AlphaProof**：AI 证明数学定理
2. **AlphaEvolve**：LLM + 进化算法发现新知识
3. **自动化发现**：AI 作为科学家
4. **未来展望**：AI 发现的方向""")

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

md("""## 1. AlphaProof

### 1.1 AI 证明数学

```
AlphaProof (DeepMind, 2024):
  用 AI 证明数学竞赛题

成就:
  - IMO 银牌级别
  - 解决了 IMO 2024 的 4/6 题
  - 包括最难的第6题

方法:
  1. 把数学问题翻译为 Lean (形式化语言)
  2. AlphaProof 生成证明步骤
  3. Lean 验证器检查每步
  4. 如果验证失败 → 重新生成

关键:
  - 形式化: Lean 保证证明正确
  - 搜索: 在证明空间搜索
  - RL: 强化学习优化证明策略
```

> 💡 AlphaProof 的关键：形式化语言保证正确性，AI 负责搜索证明路径。""")

md("""## 2. AlphaEvolve

### 2.1 LLM + 进化算法

```
AlphaEvolve:
  LLM 生成候选 → 评估 → 进化 → 发现新知识

流程:
  1. LLM 生成候选解 (代码/算法/公式)
  2. 自动评估 (运行/证明/测试)
  3. 选优秀的 → 交叉/变异
  4. LLM 基于优秀候选生成新一代
  5. 重复 → 涌现新发现

vs 纯 LLM:
  LLM: 一次生成 → 可能错
  AlphaEvolve: 迭代进化 → 逐步改进

vs 纯进化:
  进化: 随机变异 → 搜索效率低
  AlphaEvolve: LLM 变异 → 有方向性
```

### 2.2 成果

```
AlphaEvolve 发现:
  - 新的矩阵乘法算法
  - 新的数学恒等式
  - 更好的张量分解
  → 超过人类已知最优
```

> 💡 AlphaEvolve = LLM 的创造力 + 进化的筛选力——两者结合产生新知识。""")

code("""# AlphaEvolve 简化实现
class AlphaEvolve:
    def __init__(self, population_size=20, elite_ratio=0.2):
        self.pop_size = population_size
        self.elite_ratio = elite_ratio
        self.population = []
        self.history = []

    def initialize(self, generator_fn, n):
        for _ in range(n):
            candidate = generator_fn()
            fitness = self.evaluate(candidate)
            self.population.append((candidate, fitness))

    def evaluate(self, candidate):
        # 模拟评估 (实际是运行/证明/测试)
        return np.random.uniform(0, 1)

    def evolve_step(self, generator_fn):
        # 排序
        self.population.sort(key=lambda x: -x[1])
        n_elite = int(self.pop_size * self.elite_ratio)
        elite = self.population[:n_elite]

        # 记录最佳
        best = self.population[0]
        self.history.append(best[1])

        # 新一代: 精英 + LLM 变异
        new_pop = list(elite)
        while len(new_pop) < self.pop_size:
            parent = elite[np.random.randint(len(elite))]
            child = generator_fn(parent)  # LLM 基于父代生成
            fitness = self.evaluate(child)
            new_pop.append((child, fitness))

        self.population = new_pop
        return best

    def run(self, generator_fn, n_generations=20):
        self.initialize(generator_fn, self.pop_size)
        for gen in range(n_generations):
            best = self.evolve_step(generator_fn)
        return self.history

# 模拟 LLM 生成器
def llm_generator(parent=None):
    if parent is None:
        return f"算法_{np.random.randint(1000)}"
    else:
        return f"改进_{parent[0]}"

# 运行 AlphaEvolve
ae = AlphaEvolve(population_size=20, elite_ratio=0.2)
history = ae.run(llm_generator, n_generations=30)

print(f"7. AlphaEvolve 进化:")
print(f"  初始最佳: {history[0]:.4f}")
print(f"  最终最佳: {history[-1]:.4f}")
print(f"  改进: {history[-1]/history[0]:.2f}x")

# 可视化
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history, 'b-', linewidth=2)
ax.set_xlabel('进化代数', fontsize=12)
ax.set_ylabel('最佳适应度', fontsize=12)
ax.set_title('AlphaEvolve: LLM + 进化算法', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_alphaevolve.png', bbox_inches='tight')
plt.show()""")

md("""## 3. 自动化科学发现

### 3.1 AI 作为科学家

```
AI 科学发现流程:
  1. 观察数据
  2. 提出假设 (LLM 生成)
  3. 设计实验 (LLM 规划)
  4. 执行实验 (自动/模拟)
  5. 分析结果
  6. 修正假设
  7. 重复 → 发现

代表系统:
  - AlphaFold: 发现蛋白质结构
  - GNoME: 发现新材料
  - AlphaProof: 发现数学证明
  - AlphaEvolve: 发现新算法
```

### 3.2 发现 vs 优化

```
优化: 在已知空间找最优
  - 超参调优
  - 结构搜索

发现: 找到新知识/新概念
  - 新定理
  - 新算法
  - 新材料
  - 新药物

→ 发现更难，需要创造力
→ LLM 的创造力 + 验证器 = 自动化发现
```""")

code("""# 自动化发现系统对比
fig, ax = plt.subplots(figsize=(12, 6))

systems = ['AlphaFold', 'GNoME', 'AlphaProof', 'AlphaEvolve', '传统科学']
discovery_rate = [9, 9, 7, 8, 2]  # 发现速度
verification = [10, 8, 10, 9, 10]  # 验证可靠性
creativity = [5, 5, 7, 9, 8]  # 创造力

x = np.arange(len(systems))
width = 0.25

ax.bar(x - width, discovery_rate, width, label='发现速度', color='steelblue', alpha=0.8)
ax.bar(x, verification, width, label='验证可靠性', color='forestgreen', alpha=0.8)
ax.bar(x + width, creativity, width, label='创造力', color='coral', alpha=0.8)

ax.set_xlabel('系统', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('自动化科学发现系统', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(systems, fontsize=10, rotation=15)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_auto_discovery.png', bbox_inches='tight')
plt.show()
print("AI 系统发现速度远超传统; AlphaEvolve 创造力最强; 传统验证最可靠。")""")

md("""## 4. 未来展望

### 4.1 AI 发现的方向

```
短期 (1-2年):
  - 更多数学定理
  - 更多新材料
  - 更高效的算法

中期 (3-5年):
  - 新物理理论
  - 新生物机制
  - 新药物靶点

长期 (5-10年):
  - AI 自主研究
  -$新科学范式
  - 人机协作发现
```

### 4.2-挑战

```
1. 验证: 发现的东西是否正确?
2. 可解释: 为什么这个发现有效?
3. 创新: 真正的新发现 vs 已知的变体
4. 评估: 如何衡量"发现"的价值?
```""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| AlphaProof (AI 数学证明) | ✅ |
| AlphaEvolve (LLM + 进化) | ✅ |
| 自动化科学发现 | ✅ |
| 发现 vs 优化 | ✅ |

### 核心 takeaway
> **AI 正在从"解决问题"到"发现知识"**——AlphaProof 用形式化保证正确，AlphaEvolve 用 LLM+进化产生/创造力。自动化发现是 AI 的下一个前沿。

### 🔗 下一板块
**`69_sae_interpretability.ipynb`** — 稀疏自编码器、机制可解释性（进入板块十二）

---

> 💬 **板块十一(AI for Science & 自动化发现)100%完成 (6/6)。**""")

output_path = "notebooks/68_automated_discovery.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")