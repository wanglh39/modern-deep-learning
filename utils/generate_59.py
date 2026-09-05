# 生成 59_computer_use.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 59 — Computer Use 与 Browser Use

> 🔥 让 AI 像人一样操作电脑——点击、输入、滚动。

## 本章你将掌握

1. **Computer Use**：AI 操作电脑
2. **Browser Use**：AI 浏览网页
3. **动作空间**：点击/输入/滚动
4. **视觉理解**：截图 + 理解
5. **应用场景**：自动化办公""")

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

md("""## 1. Computer Use

### 1.1 核心思想

```
Computer Use:
  让 AI 像人一样操作电脑
  - 看屏幕 (截图)
  - 决定动作 (点击/输入)
  - 执行动作
  - 观察结果
  - 循环直到完成

vs 传统自动化:
  传统: 脚本/API (需要接口)
  Computer Use: 视觉+操作 (像人一样)

vs RPA:
  RPA: 固定流程 (规则驱动)
  Computer Use: 自适应 (AI 驱动)
```

### 1.2 动作空间

```
Computer Use 动作:
  1. click(x, y): 点击坐标
  2. type(text): 输入文本
  3. scroll(direction): 滚动
  4. key(key): 按键
  5. drag(src, dst): 拖拽
  6. wait(seconds): 等待
  7. screenshot(): 截图

→ 与人类操作电脑的动作相同
```

> 💡 Computer Use 是 AI 的"手"——通过视觉理解屏幕，通过动作操作电脑。""")

md("""## 2. Computer Use 架构

### 2.1 循环

```
while not done:
    1. 截图 → 视觉输入
    2. LLM(截图 + 任务 + 历史) → 动作
    3. 执行动作
    4. 等待界面更新
    5. 检查是否完成

关键:
  - 视觉理解: LLM 需要理解截图
  - 动作映射: 把 LLM 输出转为实际操作
  - 状态跟踪: 知道当前在哪、做了什么
```""")

code("""# Computer Use 简化实现
class ComputerUse:
    def __init__(self):
        self.screen = "桌面"
        self.history = []
        self.task_complete = False

    def screenshot(self):
        # 模拟截图
        return f"截图: {self.screen}"

    def execute(self, action, **kwargs):
        # 执行动作
        if action == "click":
            x, y = kwargs.get('x'), kwargs.get('y')
            result = f"点击 ({x}, {y})"
            if "浏览器" in self.screen:
                self.screen = "浏览器页面"
            elif "搜索框" in self.screen:
                self.screen = "搜索结果"
        elif action == "type":
            text = kwargs.get('text')
            result = f"输入 '{text}'"
        elif action == "key":
            key = kwargs.get('key')
            result = f"按键 {key}"
            if key == "Enter":
                self.screen = "搜索结果"
        elif action == "scroll":
            result = f"滚动 {kwargs.get('direction')}"
        else:
            result = f"执行 {action}"

        self.history.append(result)
        return result

    def decide(self, task, screenshot):
        # 模拟 LLM 决策
        if "打开" in task and "浏览器" not in self.screen:
            return "click", {"x": 100, "y": 50}  # 点击浏览器图标
        elif "搜索" in task and "搜索框" not in self.screen:
            return "click", {"x": 200, "y": 100}  # 点击搜索框
        elif "搜索" in task:
            return "type", {"text": task.split("搜索")[1]}
        elif "搜索结果" in self.screen:
            return "key", {"key": "Enter"}
        else:
            self.task_complete = True
            return "done", {}

    def run(self, task, max_steps=10):
        print(f"任务: {task}")
        print("=" * 50)

        for step in range(max_steps):
            # 1. 截图
            screenshot = self.screenshot()
            print(f"\\n步骤 {step+1}: {screenshot}")

            # 2. 决策
            action, kwargs = self.decide(task, screenshot)
            if action == "done":
                print("  → 任务完成!")
                break

            # 3. 执行
            result = self.execute(action, **kwargs)
            print(f"  → {result}")

        return self.history

# 演示
computer = ComputerUse()
history = computer.run("打开浏览器搜索AI")""")

md("""## 3. Browser Use

### 3.1 专门针对浏览器

```
Browser Use:
  Computer Use 的浏览器特化
  - 更高效 (直接操作 DOM)
  - 更可靠 (不依赖视觉)
  - 更快 (不需要截图)

两种模式:
  1. 视觉模式: 截图 + 点击坐标
     → 通用但慢
  
  2. DOM 模式: 解析 HTML + 操作元素
     → 快但需要适配
```

### 3.2 Browser Use 工具

```
浏览器专用工具:
  - navigate(url): 导航
  - click_element(selector): 点击元素
  - fill_form(selector, value): 填表
  - extract_text(): 提取文本
  - scroll_page(): 滚动
  - get_links(): 获取链接
  - take_screenshot(): 截图

→ 比通用 Computer Use 更精确
```""")

code("""# Browser Use 实现
class BrowserUse:
    def __init__(self):
        self.url = "about:blank"
        self.page_content = ""
        self.history = []

    def navigate(self, url):
        self.url = url
        self.page_content = f"页面内容: {url} 的首页"
        self.history.append(f"导航到 {url}")
        return self.page_content

    def click(self, selector):
        self.history.append(f"点击 {selector}")
        return f"已点击 {selector}"

    def fill(self, selector, value):
        self.history.append(f"填入 {selector} = {value}")
        return f"已填入"

    def extract(self):
        self.history.append("提取文本")
        return f"提取: {self.page_content}"

    def run(self, task):
        # 模拟浏览器自动化
        if "搜索" in task:
            self.navigate("google.com")
            self.fill("搜索框", task)
            self.click("搜索按钮")
            results = self.extract()
            return results

agent = BrowserUse()
result = agent.run("搜索AI最新进展")
print("Browser Use:")
print("=" * 50)
for action in agent.history:
    print(f"  {action}")
print(f"\\n结果: {result}")""")

md("""## 4. 视觉理解

### 4.1 截图理解

```
Computer Use 的关键: LLM 理解截图

挑战:
  - 分辨率: 截图可能很大
  - UI 元素: 按钮、输入框、菜单
  - 文本: OCR + 理解
  - 布局: 元素位置关系

解决:
  - 多分辨率: 先看缩略图，再看细节
  - 元素检测: 识别可交互元素
  - 坐标映射: 把"点击按钮"转为坐标
```""")

code("""# 视觉理解模拟
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 模拟屏幕
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.add_patch(plt.Rectangle((0, 5), 10, 1, color='gray', alpha=0.3))  # 标题栏
ax.add_patch(plt.Rectangle((1, 4), 2, 0.5, color='steelblue', alpha=0.7))  # 按钮
ax.add_patch(plt.Rectangle((4, 4), 4, 0.5, color='white', edgecolor='black'))  # 输入框
ax.text(2, 5.5, '浏览器', ha='center', fontsize=10)
ax.text(2, 4.25, '搜索', ha='center', fontsize=9, color='white')
ax.text(6, 4.25, '输入搜索词...', ha='center', fontsize=9, color='gray')
ax.set_title('屏幕截图', fontsize=13, fontweight='bold')
ax.grid(alpha=0.2)

# 检测到的元素
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax.add_patch(plt.Rectangle((1, 4), 2, 0.5, fill=False, edgecolor='red', linewidth=2))
ax.add_patch(plt.Rectangle((4, 4), 4, 0.5, fill=False, edgecolor='blue', linewidth=2))
ax.text(2, 4.7, '按钮[click]', ha='center', fontsize=8, color='red')
ax.text(6, 4.7, '输入框[type]', ha='center', fontsize=8, color='blue')
ax.set_title('检测到的可交互元素', fontsize=13, fontweight='bold')
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('notebooks/fig_computer_use.png', bbox_inches='tight')
plt.show()
print("Computer Use: 截图 → 检测元素 → 决定动作 → 执行。")""")

md("""## 5. 应用场景

### 5.1 典型应用

```
1. 自动化办公
   - 填表单、发邮件
   - 数据录入
   - 文档处理

2. 网页自动化
   - 网页搜索
   - 信息采集
   - 表单提交

3. 软件测试
   - UI 测试
   - 端到端测试
   - 回归测试

4. 辅助操作
   - 帮残障人士操作电脑
   - 教学演示
   - 远程协助
```""")

code("""# Computer Use vs Browser Use 对比
fig, ax = plt.subplots(figsize=(10, 6))

aspects = ['通用性', '速度', '可靠性', '开发难度', '适用范围']
computer_use = [9, 4, 5, 8, 9]
browser_use = [5, 8, 8, 5, 5]

x = np.arange(len(aspects))
width = 0.35

ax.bar(x - width/2, computer_use, width, label='Computer Use', color='steelblue', alpha=0.8)
ax.bar(x + width/2, browser_use, width, label='Browser Use', color='coral', alpha=0.8)

ax.set_xlabel('方面', fontsize=12)
ax.set_ylabel('评分 (1-10)', fontsize=12)
ax.set_title('Computer Use vs Browser Use', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(aspects, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/fig_computer_vs_browser.png', bbox_inches='tight')
plt.show()
print("Computer Use 更通用; Browser Use 更快更可靠。")""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| Computer Use (操作电脑) | ✅ |
| Browser Use (浏览器自动化) | ✅ |
| 动作空间 | ✅ |
| 视觉理解 | ✅ |
| 应用场景 | ✅ |

### 核心 takeaway
> **Computer Use 让 AI 像人一样操作电脑**——截图理解+动作执行。Browser Use 是浏览器特化版，更快更可靠。

### 🔗 下一章
**`60_agent_evaluation.ipynb`** — Agent 评估、benchmark

---

> 💬 **板块九(Agent 与系统)进行中 (8/9)。**""")

output_path = "notebooks/59_computer_use.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")