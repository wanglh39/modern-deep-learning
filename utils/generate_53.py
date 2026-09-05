# 生成 53_tool_use_mcp.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 53 — 工具使用与 MCP 协议

> 🔥 Function Calling 让 LLM 能"动手"，MCP 让工具接入标准化。

## 本章你将掌握

1. **Function Calling**：LLM 调用函数的机制
2. **工具设计**：好的工具描述
3. **MCP 协议**：Model Context Protocol
4. **工具编排**：多工具组合""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
print("环境准备完成 ✅")""")

md("""## 1. Function Calling

### 1.1 核心机制

```
Function Calling 流程:
  1. 用户定义可用函数 (名称、描述、参数)
  2. LLM 收到 prompt + 函数定义
  3. LLM 决定是否调用函数
  4. 如果调用: LLM 输出函数名 + 参数 (JSON)
  5. 系统执行函数
  6. 把结果返回给 LLM
  7. LLM 基于结果继续生成

关键: LLM 不直接执行函数
      LLM 只输出"要调用什么"
      系统负责执行
```

### 1.2 函数定义格式

```json
{
  "name": "get_weather",
  "description": "获取指定城市的天气",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "城市名"},
      "unit": {"type": "string", "enum": ["C", "F"]}
    },
    "required": ["city"]
  }
}
```

> 💡 Function Calling 的关键：好的函数描述让 LLM 知道何时、如何调用。""")

code("""# Function Calling 实现
class FunctionCaller:
    def __init__(self):
        self.functions = {}

    def register(self, name, description, func, params_schema):
        self.functions[name] = {
            'description': description,
            'func': func,
            'params': params_schema,
        }

    def get_schema(self):
        schemas = []
        for name, info in self.functions.items():
            schemas.append({
                'name': name,
                'description': info['description'],
                'parameters': info['params'],
            })
        return schemas

    def call(self, name, arguments):
        if name not in self.functions:
            return f"错误: 未知函数 {name}"
        try:
            result = self.functions[name]['func'](**arguments)
            return result
        except Exception as e:
            return f"错误: {e}"

# 模拟 LLM 决策
def llm_decide(prompt, available_functions):
    # 模拟 LLM 选择函数 (用简单规则)
    if "天气" in prompt:
        return {"name": "get_weather", "arguments": {"city": "北京"}}
    elif "计算" in prompt:
        return {"name": "calculate", "arguments": {"expr": "2+3*4"}}
    elif "搜索" in prompt:
        return {"name": "search", "arguments": {"query": prompt}}
    else:
        return None

# 注册函数
fc = FunctionCaller()

fc.register(
    "get_weather",
    "获取指定城市的天气",
    lambda city: f"{city}今天25度，晴",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
)

fc.register(
    "calculate",
    "计算数学表达式",
    lambda expr: str(eval(expr)),
    {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}
)

fc.register(
    "search",
    "搜索信息",
    lambda query: f"搜索结果: 关于'{query}'的信息",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
)

# 测试
prompts = ["北京天气怎么样？", "计算 2+3*4", "搜索AI最新进展"]

for prompt in prompts:
    print(f"用户: {prompt}")
    decision = llm_decide(prompt, fc.get_schema())
    if decision:
        result = fc.call(decision['name'], decision['arguments'])
        print(f"  LLM 调用: {decision['name']}({decision['arguments']})")
        print(f"  结果: {result}\\n")
    else:
        print(f"  LLM 直接回答\\n")""")

md("""## 2. MCP (Model Context Protocol)

### 2.1 为什么需要 MCP？

```
问题: 每个 Agent 框架有自己的工具接口
  - LangChain 工具 ≠ AutoGen 工具 ≠ CrewAI 工具
  - 工具开发者要为每个框架写适配
  - 重复劳动

MCP (Model Context Protocol):
  - Anthropic 提出的开放标准
  - 统一工具/资源/Prompt 的接口
  - 一次实现，到处使用

类比:
  - MCP 之于 Agent = USB 之于电脑
  - 标准化接口 → 即插即用
```

### 2.2 MCP 架构

```
MCP 架构:
  MCP Host (如 Claude Desktop)
    ↓
  MCP Client (协议客户端)
    ↓ (JSON-RPC)
  MCP Server (工具提供者)
    ↓
  工具/资源/Prompts

MCP 三大原语:
  1. Tools: 可执行函数
  2. Resources: 可读数据
  3. Prompts: 可复用提示模板
```

> 💡 MCP 是 Agent 的"USB 标准"——一次实现，到处使用，即插即用。""")

code("""# MCP 协议简化实现
class MCPServer:
    def __init__(self, name):
        self.name = name
        self.tools = {}
        self.resources = {}
        self.prompts = {}

    def register_tool(self, name, description, handler, input_schema):
        self.tools[name] = {
            'description': description,
            'handler': handler,
            'inputSchema': input_schema,
        }

    def register_resource(self, uri, description, handler):
        self.resources[uri] = {
            'description': description,
            'handler': handler,
        }

    def list_tools(self):
        return [{"name": k, "description": v['description'],
                 "inputSchema": v['inputSchema']}
                for k, v in self.tools.items()]

    def call_tool(self, name, arguments):
        if name not in self.tools:
            return {"error": f"未知工具: {name}"}
        return {"result": self.tools[name]['handler'](**arguments)}

    def read_resource(self, uri):
        if uri not in self.resources:
            return {"error": f"未知资源: {uri}"}
        return {"content": self.resources[uri]['handler']()}

class MCPClient:
    def __init__(self, server):
        self.server = server

    def list_tools(self):
        return self.server.list_tools()

    def call_tool(self, name, arguments):
        return self.server.call_tool(name, arguments)

    def read_resource(self, uri):
        return self.server.read_resource(uri)

# 创建 MCP Server
server = MCPServer("weather-server")

server.register_tool(
    "get_forecast",
    "获取天气预报",
    lambda city, days=3: f"{city}未来{days}天: 晴/多云/雨",
    {"type": "object", "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer", "default": 3}
    }, "required": ["city"]}
)

server.register_resource(
    "weather://cities",
    "支持的城市列表",
    lambda: "北京, 上海, 广州, 深圳"
)

# MCP Client 使用
client = MCPClient(server)

print("MCP 协议演示:")
print("=" * 50)
print(f"\\n1. 列出工具:")
tools = client.list_tools()
print(json.dumps(tools, indent=2, ensure_ascii=False))

print(f"\\n2. 调用工具:")
result = client.call_tool("get_forecast", {"city": "北京", "days": 5})
print(f"   get_forecast(北京, 5) → {result}")

print(f"\\n3. 读取资源:")
resource = client.read_resource("weather://cities")
print(f"   weather://cities → {resource}")""")

md("""## 3. 工具设计最佳实践

### 3.1 好工具的特征

```
1. 清晰的描述
   - LLM 靠描述决定何时调用
   - 描述要说明: 做什么 + 何时用 + 参数含义

2. 合理的粒度
   - 太细: LLM 要调很多次
   - 太粗: 灵活性不够
   - 找平衡点

3. 明确的参数
   - 类型、范围、默认值
   - required vs optional

4. 错误处理
   - 返回有意义的错误信息
   - 让 LLM 能理解并修正

5. 幂等性
   - 同样输入 → 同样输出
   - 避免副作用
```

### 3.2 工具描述示例

```
好的描述:
  "搜索网页并返回前5个结果。
   当需要获取最新信息或查找事实时使用。
   query: 搜索关键词，不超过100字符。"

差的描述:
  "搜索"  # 太简略，LLM 不知道何时用
```""")

code("""# 工具编排: 多工具组合
class ToolOrchestrator:
    def __init__(self, tools):
        self.tools = tools

    def execute_plan(self, plan):
        # plan: [(tool_name, args), ...]
        results = []
        context = {}

        for step_name, tool_name, args in plan:
            if tool_name not in self.tools:
                results.append((step_name, f"未知工具: {tool_name}"))
                continue

            # 替换参数中的上下文引用
            resolved_args = {}
            for k, v in args.items():
                if isinstance(v, str) and v.startswith("$"):
                    resolved_args[k] = context.get(v[1:], v)
                else:
                    resolved_args[k] = v

            result = self.tools[tool_name](**resolved_args)
            context[step_name] = result
            results.append((step_name, result))

        return results, context

# 定义工具
tools = {
    'search': lambda query: f"找到: {query}的相关信息",
    'extract': lambda text: f"提取关键点: {text[:20]}...",
    'summarize': lambda text: f"摘要: {text[:15]}",
    'translate': lambda text, lang="en": f"翻译({lang}): {text[:10]}",
}

orchestrator = ToolOrchestrator(tools)

# 执行计划: 搜索 → 提取 → 摘要 → 翻译
plan = [
    ("search_result", "search", {"query": "AI最新进展"}),
    ("extracted", "extract", {"text": "$search_result"}),
    ("summary", "summarize", {"text": "$extracted"}),
    ("translated", "translate", {"text": "$summary", "lang": "en"}),
]

print("工具编排:")
print("=" * 50)
results, context = orchestrator.execute_plan(plan)
for step_name, result in results:
    print(f"  {step_name}: {result}")""")

md("""## 4. 并行工具调用

### 4.1 并行 vs 串行

```
串行调用:
  search(A) → search(B) → search(C)
  时间: 3t

并行调用:
  search(A) | search(B) | search(C)
  时间: t (3倍加速)

现代 LLM 支持并行 Function Calling:
  一次输出多个函数调用
  系统并行执行
  等所有结果返回后继续
```""")

code("""# 并行工具调用
import time

class ParallelToolCaller:
    def __init__(self, tools):
        self.tools = tools

    def call_parallel(self, calls):
        # 模拟并行调用 (实际用 asyncio)
        results = []
        for name, args in calls:
            if name in self.tools:
                results.append((name, self.tools[name](**args)))
            else:
                results.append((name, f"未知工具: {name}"))
        return results

tools = {
    'search': lambda query: f"搜索结果: {query}",
    'weather': lambda city: f"{city}: 25度",
    'time': lambda: "14:30",
}

caller = ParallelToolCaller(tools)

# 并行调用多个工具
parallel_calls = [
    ("search", {"query": "AI新闻"}),
    ("weather", {"city": "北京"}),
    ("time", {}),
]

print("并行工具调用:")
results = caller.call_parallel(parallel_calls)
for name, result in results:
    print(f"  {name}: {result}")
print("\\n所有调用并行执行 → 总时间 ≈ 最慢的工具")""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| Function Calling 机制 | ✅ |
| MCP 协议 (标准化) | ✅ |
| 工具设计最佳实践 | ✅ |
| 工具编排 | ✅ |
| 并行工具调用 | ✅ |

### 核心 takeaway
> **Function Calling 让 LLM 能行动，MCP 让工具接入标准化**——好的工具描述是关键，MCP 是 Agent 的"USB 标准"，并行调用提升效率。

### 🔗 下一章
**`54_context_engineering.ipynb`** — Context Loop/Harness、context 压缩/路由

---

> 💬 **板块九(Agent 与系统)进行中 (2/9)。**""")

output_path = "notebooks/53_tool_use_mcp.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")