# 生成 46_edge_deployment.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 46 — 边缘部署：WebGPU 与5与本地部署

> 🔥 让 LLM 在浏览器和手机上运行——WebGPU、ONNX、llama.cpp。

## 本章你将掌握

1. **WebGPU**：浏览器中的 GPU
2. **llama.cpp**：C++ 推理引擎
3. **ONNX**：跨平台部署
4. **移动端**：手机上的 LLM""")

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

md("""## 1. WebGPU：浏览器中的 GPU

### 1.1 WebGPU API

```
WebGPU: 浏览器暴露 GPU 计算能力
  → JavaScript 可以直接用 GPU
  → 在浏览器中运行 LLM!

流程:
  1. 加载模型 (GGUF 格式)
  2. WebGPU 创建 buffer
  3. WGSL 着色器做矩阵乘法
  4. 逐 token 生成
```

### 1.2 代表项目

- **webllm**: WebGPU 上的 LLM 推理
- **transformers.js**: H>JS 版 HuggingFace
- **MediaPipe LLM**: Google 的方案

### 1.3 性能

```
7B 模型在浏览器:
  - 加载: ~4GB 下载 (一次)
  - 推理: ~10-20 token/s (取决于 GPU)
  - 显存: 用浏览器 GPU

→ 不需要服务器 → 隐私 + 离线
```

> 💡 WebGPU 让 LLM 在浏览器中运行——无需服务器，完全离线，隐私友好。""")

md("""## 2. llama.cpp

### 2.1 核心思想

用纯 C++ 实现 LLM 推理，极致优化：

```
llama.cpp:
  - GGUF 模型格式 (量化)
  - C++ 实现, 无依赖
  - CPU/GPU 后端
  - INT4/INT8 量化

→ 在 MacBook 上跑 70B 模型!
```

### 2.2 优势

- **无依赖**：纯 C++，编译即用
- **CPU 推理**：不需要 GPU
- **量化**：INT4 让大模型在小内存运行
- **跨平台**：Mac/Windows/Linux/手机

### 2.3 性能

```
7B INT4 在 MacBook M2:
  - 内存: ~4GB
  - 速度: ~20 token/s

70B INT4 在 MacBook M2 Max:
  - 内存: ~40GB
  - 速度: ~5 token/s
```

> 💡 llama.cpp 是本地部署的标配——纯 C++，无依赖，CPU/GPU 通吃。""")

code("""# 部署方案对比
methods = ['服务器\\n(vLLM)', 'llama.cpp\\n(本地)', 'WebGPU\\n(浏览器)', 'ONNX\\n(跨平台)', 'MLC\\n(手机)']
speeds = [100, 20, 15, 10, 8]  # token/s
privacy = [50, 100, 100, 100, 100]  # 隐私 (%)
ease = [80, 60, 90, 50, 40]  # 易用性 (%)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(methods, speeds, color='steelblue', alpha=0.8)
axes[0].set_ylabel('速度 (token/s)'); axes[0].set_title('推理速度')

axes[1].bar(methods, privacy, color='forestgreen', alpha=0.8)
axes[1].set_ylabel('隐私 (%)'); axes[1].set_title('隐私性')

axes[2].bar(methods, ease, color='coral', alpha=0.8)
axes[2].set_ylabel('易用性 (%)'); axes[2].set_title('部署易用性')

for ax in axes:
    ax.set_xticklabels(methods, fontsize=8, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('notebooks/fig_edge_deployment.png', bbox_inches='tight')
plt.show()
print("服务器最快; 本地/浏览器隐私最好; WebGPU 最易用。")""")

md("""## 3. ONNX：跨平台

### 3.1 思想

```
PyTorch → ONNX → 任意平台
  - Windows/Linux/Mac
  - CPU/GPU/NPU
  - 浏览器 (onnxruntime-web)
```

### 3.2 流程

```
1. PyTorch 训练模型
2. torch.onnx.export → ONNX 文件
3. onnxruntime 推理

→ 一次导出, 到处运行
```

> 💡 ONNX 是跨平台部署的标准——一次导出，到处运行。

## 4. 移动端

### 4.1 方案

- **MLC-LLM**: iOS/Android 上的 LLM
- **MediaPipe**: Google 的移动端方案
- **CoreML**: Apple 的方案
- **TensorFlow Lite**: Google 的轻量推理

### 4.2 挑战

```
手机限制:
  - 内存: 4-8GB → 只能跑 1-3B 模型
  - 功耗: 持续推理耗电快
  - 存储: 模型文件占用大

→ 量化 + 蒸馏是关键
```

> 💡 移动端 LLM 是未来方向——隐私、离线、低延迟。
# 1-3B 量化模型在手机上可行。""")

md("""## 5. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| WebGPU 浏览器推理 | ✅ |
| llama.cpp 本地部署 | ✅ |
| ONNX 跨平台 | ✅ |
| 移动端部署 | ✅ |

### 核心 takeaway
> **边缘部署让 LLM 无处不在**——浏览器、手机、本地。
# llama.cpp + 量化是本地部署的黄金组合，WebGPU 是浏览器方向。

### 🔗 下一板块
**`47_alignment_safety.ipynb`** — 对齐、安全与评估（进入板块八）

---

> 💬 **板块七(效率与部署)完结。**""")

output_path = "notebooks/46_edge_deployment.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")