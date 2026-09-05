# 学习指南：如何使用本项目

> 本文档帮助你理解 73 个 notebook 的定位、简化边界，以及如何补充真实工程能力。
> 最后更新：2026-09-05

---

## 一、本项目是什么

一套 **73 个 Jupyter notebook**，覆盖现代深度学习全栈（MLP → Transformer → LLM → Agent → AI for Science），注重**算法原理**（70%）兼顾工程（30%）。

**核心定位：概念地图，不是地形本身。**

每个 notebook 用最小可运行代码，把数学公式变成 PyTorch 代码，让你在 CPU 上秒级跑通、建立对整个技术栈的全局认知。

---

## 二、本项目不是什么

- **不是工程级实现参考**：模型规模玩具级（d_model=32~128，2~6 层），真实模型差 5-6 个数量级
- **不是真实数据实验**：95%+ 用随机数据，看不到真实数据上的训练动态
- **不是完整复现**：部分 notebook（如 Stable Diffusion、AlphaFold）只演示架构骨架，无训练循环
- **不是评估教程**：几乎没有量化指标（FID/Perplexity/Accuracy），只看 loss 曲线

**一句话：跑通这些 notebook ≠ 能在真实场景复现。** 这是"我懂了概念"的第一步，不是终点。

---

## 三、五类简化清单

### 1. 数据：随机生成，非真实数据集
- 95%+ 的 notebook 用 `torch.randn` / `np.random` 生成数据
- 0 个用 ImageNet / wikipedia / 真实蛋白质结构
- 仅 2 个用 sklearn 8x8 MNIST（`load_digits()`）
- **影响**：看不到真实数据上的过拟合、泛化、数据增强效果

### 2. 模型规模：玩具级
| 维度 | 本项目 | 真实规模 |
|------|--------|---------|
| d_model | 16–384 | 4096–12288 |
| n_layers | 2–6 | 32–96 |
| 参数量 | 数千～数十万 | 7B–175B |
- **影响**：看不到大模型特有的现象（scaling law 实测、涌现能力、训练不稳定性）

### 3. 训练规模：极短
- epoch 典型 50–200，最大 2000（仅 2D 玩具数据）
- 数据量几十到几千条，单 batch 为主
- 无梯度累积、无混合精度、无分布式
- **影响**：缺乏对真实训练动态的直觉（loss spike、梯度爆炸、学习率调度）

### 4. 算法：核心在，工程全砍
- **保留**：核心数学公式（PPO 裁剪、DDPM 闭式解、Cross-attention、Bradley-Terry 损失等）
- **砍掉**：U-Net→MLP、真实分词器→空格切词、数据加载 pipeline、分布式/量化/部署
- 部分核心创新被省略：AlphaFold 的三角更新、Stable Diffusion 的 U-Net、RLHF 作用于真实 LM
- **影响**：部分算法的精髓缺失

### 5. 评估：几乎无量化指标
- 73 个中仅 1 个有 train_test_split（且玩具数据）
- 无 FID / Perplexity / Accuracy / BLEU
- 评估方式：画 loss 曲线 + 打印几个样例
- **影响**：无法判断"模型到底行不行"

---

## 四、正确使用方式

### 阶段一：用本项目建立全局认知（你已在这里）
- 按板块顺序读 notebook，理解每个概念的"是什么、为什么、怎么做"
- 重点看 markdown 讲解和公式→代码的映射
- 跑通 code cell，改改参数看变化

### 阶段二：每个板块补一个真实实现
见下方"补充学习路径"表。每个板块选 1-2 个真实项目读源码或跑通。

### 阶段三：在真实数据上完整跑一次
哪怕只是 CIFAR-10 + ResNet，也要经历：数据加载 → 训练 → 验证 → 测试 → 调参 → 部署的完整闭环。

---

## 五、补充学习路径（每板块推荐真实实现）

| 板块 | 本项目 notebook | 下一步该读/该跑的真实实现 |
|------|----------------|------------------------|
| 一、基础与范式 | 01–08 | Karpathy `micrograd`（手写 autograd）；`timm` 库（ViT/ResNet 实现） |
| 二、训练范式 | 09–18 | Karpathy `nanoGPT`（GPT-2 从零训练）；HuggingFace `transformers` Trainer 源码 |
| 三、推理模型 | 19–22 | OpenAI o1 技术报告；DeepSeek-R1 论文 + 开源权重 |
| 四、多模态 | 23–29 | `transformers` CLIP/BLIP 源码；LLaVA 项目；Isaac Gym 官方教程 |
| 五、表征/检索 | 30–32 | `sentence-transformers` 库；FAISS 官方教程；ColBERT 源码 |
| 六、生成模型 | 33–40 | HuggingFace `diffusers` 源码（SD 完整 pipeline）；`stable-diffusion-webui` |
| 七、效率部署 | 41–46 | `vllm` 源码；`llama.cpp`；HuggingFace `optimum` 量化教程 |
| 八、对齐安全 | 47–51 | HuggingFace `trl` 库（RLHF/DPO 实现）；Anthropic Constitutional AI 论文 |
| 九、Agent | 52–60 | LangGraph / AutoGen 官方 tutorial；`anthropic-cookbook` tool use 示例 |
| 十、时序 | 61–62 | `transformers` TimesFM；HuggingFace 时序模型教程 |
| 十一、AI4Science | 63–68 | DeepMind AlphaFold 开源代码；`unimol` 分子建模库 |
| 十二、可解释性 | 69–71 | Anthropic SAE 论文 + `sae_lens` 库；`transformer-lens` |
| 拓展 | 72–73 | DeepMind AlphaZero 论文；PyG (PyTorch Geometric) 官方教程 |

---

## 六、推荐的真实数据集练手项目

如果想补"真实数据 + 完整训练闭环"的经验，推荐：

1. **CIFAR-10 + ResNet**（1 小时）：torchvision 数据加载 → 训练 → 测试 → 混淆矩阵
2. **AG News + 小 BERT 微调**（2 小时）：HuggingFace `datasets` + `Trainer` → 文本分类
3. **tinyshakespeare + nanoGPT**（3 小时）：Karpathy 的经典练手项目，真实文本训练字符级 GPT
4. **Kaggle 入门比赛**：真实数据 + 评估指标 + leaderboard，补全"模型到底行不行"的判断力

---

## 七、心态建议

- **不要产生"我懂了"的虚假信心**：跑通 notebook 只是理解了概念骨架，离工程实现还有距离
- **也不要觉得"白学了"**：全局认知地图极有价值——知道技术全貌、知道每个技术解决什么问题、知道该深入哪个方向，这本身就是巨大的学习成果
- **正确的节奏**：概念地图（本项目）→ 选定方向深入真实实现 → 在真实数据上完整跑通 → 形成工程直觉

---

> 本项目是起点，不是终点。地图已经给你了，地形要自己去走。