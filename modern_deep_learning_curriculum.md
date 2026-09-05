# 现代深度学习教学项目 — 总体框架与进度追踪

> 本文档是项目的总纲，记录设计决策、完整章节框架、进度追踪与依赖关系。
> 项目共 **73 个 Jupyter notebook**，分 **12 主线板块 + 拓展章**。
> 最后更新：2026-09-05

---

## 一、项目概述

### 定位
- **核心目标**：注重算法原理（70%），兼顾工程性能（30%），用 PyTorch 框架，历史脉络演进 + 对比穿插（经典服务现代）
- **交付形态**：Jupyter notebook 为主，每章原理讲解 + 代码 + 可视化一体，可运行可改
- **粒度原则**：一概念一 notebook，讲透而非堆砌
- **目标受众**：有 Python/C/C++ 基础，学过强化学习，希望系统掌握现代深度学习全貌的学习者

### 技术栈
- PyTorch（主框架）
- Jupyter notebook（交付形态）
- 配套可视化（matplotlib / plotly / 交互式 widget）

---

## 二、设计决策记录

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 框架 | PyTorch | 用户指定，注重算法而非造轮子 |
| 组织方式 | 历史脉络演进 + 对比穿插 | 经典模型服务现代，CNN↔ViT、RNN↔Transformer |
| 算法/工程配比 | 70% / 30% | 算法为主，工程作为落地实践章节 |
| 交付形态 | Jupyter notebook | 教学最直观，原理+代码+可视化一体 |
| 经典基础定位 | 对比穿插：经典服务现代 | 每个经典模型和现代继承者配对讲 |
| RL 定位 | 混合：主线贯穿 + 拓展章集中讲前沿 | 对齐/具身/Agent 用到时讲，AlphaGo/多智能体集中拓展 |
| 博弈论 | 不前置，就地讲 | 在 GAN/多智能体/对齐用到时讲博弈论概念 |
| 群智能 | 并入多 agent 视角 | 多 agent 协作本质有群智能思想 |
| 进化算法 | 不独立成章 | AlphaEvolve 归"AI for Science & 自动化发现"，神经进化/ES 归"非梯度优化" |
| 粒度 | 一概念一 notebook，拆细 | 70% 算法深度要求每个算法讲透 |
| RAG 归属 | Agent 板块 | RAG 本质是 agent 的检索工具 |
| 长上下文归属 | 推理模型板块 | 是推理时能力 |
| 时序/AI4Science/可解释性 | 独立成章 | 当前热门，不该降级为拓展 |
| 持续学习/联邦学习 | 砍掉 | 相对学术小众，未保留 |

---

## 三、图例说明

| 标记 | 含义 |
|------|------|
| 🧭 | RL（强化学习）贯穿处 |
| 🔥 | 当前热点（2024-2026） |
| 🪙 | 博弈论视角显式讲解 |
| 🐝 | 群智能视角 |
| `[新]` | 第二轮补充的 notebook |
| `[新2]` | 第三轮补充的 notebook |
| `[并入]` | 并入相邻 notebook 的内容 |

### 进度状态标记
| 状态 | 含义 |
|------|------|
| ⬜ 未开始 | 尚未动工 |
| 🔄 进行中 | 正在实现 |
| ✅ 已完成 | 已完成并验证 |

---

## 四、完整框架

### 板块一：基础与范式演进（8 notebooks）

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 01 | `01_mlp_backprop.ipynb` | MLP + 反向传播 + 计算图 | ✅ |
| 02 | `02_autograd_optimizers.ipynb` | 自动微分 + 优化器(SGD→Adam→AdamW) + 正则化 | ✅ |
| 03 | `03_normalization_activation.ipynb` | 归一化(BN→LN→RMSNorm) + 激活(ReLU→GELU→SwiGLU) + 初始化 `[新]` | ✅ |
| 04 | `04_positional_encoding.ipynb` | 位置编码(绝对→RoPE→ALiBi) `[新]` | ✅ |
| 05 | `05_generalization_theory.ipynb` | 泛化理论、双下降、万能近似定理 `[新2]` | ✅ |
| 06 | `06_cnn_vs_vit.ipynb` | CNN(含LeNet/VGG/ResNet) ↔ ViT 对比 | ✅ |
| 07 | `07_rnn_vs_transformer.ipynb` | RNN/LSTM ↔ Transformer 对比 | ✅ |
| 08 | `08_architecture_innovations.ipynb` | 🔥MoE/SSM-Mamba/线性注意力/液态网络 + GQA/MQA `[并入]` | ✅ |

### 板块二：现代训练范式（10 notebooks）🧭

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 09 | `09_tokenization.ipynb` | BPE/SentencePiece/Unigram/tiktoken `[新]` | ✅ |
| 10 | `10_pretraining_scaling.ipynb` | 自监督预训练(MLM/CLM/MAE) + Scaling law | ✅ |
| 11 | `11_sft_peft.ipynb` | SFT + LoRA/QLoRA + DoRA/PiSSA `[并入]` | ✅ |
| 12 | `12_rlhf_ppo.ipynb` | 🧭RLHF + PPO 底层机制 | ✅ |
| 13 | `13_dpo_family.ipynb` | 🔥DPO → GRPO → DAPO/GSPO | ✅ |
| 14 | `14_data_engineering.ipynb` | 合成数据、筛选、配比 + 自训练/自我改进 `[并入]` | ✅ |
| 15 | `15_training_techniques.ipynb` | 学习率调度/warmup/cosine/梯度累积/裁剪 `[新]` | ✅ |
| 16 | `16_model_merging.ipynb` | TIES/DARE/SLERP 模型合并 `[新2]` | ✅ |
| 17 | `17_retrieval_augmented_training.ipynb` | RETRO/kNN-LM 检索增强训练 `[新2]` | ✅ |
| 18 | `18_non_gradient_optimization.ipynb` | 神经进化/ES/NAS 非梯度优化 | ✅ |

### 板块三：推理与推理模型（4 notebooks）🔥

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 19 | `19_reasoning_models.ipynb` | o1/o3、DeepSeek-R1、test-time scaling | ✅ |
| 20 | `20_prm_process_reward.ipynb` | PRM 过程奖励模型 `[新]` | ✅ |
| 21 | `21_inference_time_methods.ipynb` | best-of-N/self-consistency/verification `[新]` | ✅ |
| 22 | `22_long_context.ipynb` | 长上下文、Ring Attention、位置编码外推 | ✅ |

### 板块四：多模态与具身（7 notebooks）🧭🔥

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 23 | `23_clip_blip.ipynb` | CLIP、BLIP 多模态对齐 | ✅ |
| 24 | `24_vlm.ipynb` | VLM、原生多模态 | ✅ |
| 25 | `25_world_models_video.ipynb` | Sora、视频生成、世界模型 | ✅ |
| 26 | `26_3d_representation.ipynb` | NeRF/3D Gaussian Splatting `[新]` | ✅ |
| 27 | `27_embodied_ai.ipynb` | 🧭VLA、RT-2、机器人 | ✅ |
| 28 | `28_embodied_simulation.ipynb` | Isaac Gym/Habitat 仿真环境 `[新]` | ✅ |
| 29 | `29_speech_audio.ipynb` | ASR、TTS、Whisper | ✅ |

### 板块五：表征学习与检索（3 notebooks）

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 30 | `30_contrastive_learning.ipynb` | SimCLR、MoCo、InfoNCE | ✅ |
| 31 | `31_non_contrastive_ssl.ipynb` | BYOL/DINO/DINOv2 非对比自监督 `[新]` | ✅ |
| 32 | `32_embedding_retrieval.ipynb` | 🔥多向量embedding、向量检索(10B规模) | ✅ |

### 板块六：生成模型（8 notebooks）

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 33 | `33_ddpm.ipynb` | DDPM、DDIM 采样 | ✅ |
| 34 | `34_stable_diffusion.ipynb` | Latent Diffusion、条件生成 | ✅ |
| 35 | `35_controlnet.ipynb` | ControlNet/IP-Adapter 条件控制 `[新]` | ✅ |
| 36 | `36_flow_matching.ipynb` | Flow Matching/Rectified Flow `[新]` | ✅ |
| 37 | `37_consistency_models.ipynb` | 一致性模型（扩散加速）`[新]` | ✅ |
| 38 | `38_dit.ipynb` | DiT 扩散 Transformer | ✅ |
| 39 | `39_vae_gan_flow.ipynb` | VAE、GAN(�纳什均衡)、Normalizing Flow | ✅ |
| 40 | `40_autoregressive_vs_diffusion.ipynb` | 自回归生成 vs 扩散生成对比 | ✅ |

### 板块七：效率与部署（6 notebooks）

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 41 | `41_distillation_quantization.ipynb` | 知识蒸馏、🔥量化感知修复(PTQ/QAT/GPTQ/AWQ) | ✅ |
| 42 | `42_pruning_moe.ipynb` | 剪枝、稀疏化、MoE 效率 | ✅ |
| 43 | `43_paged_attention_serving.ipynb` | PagedAttention + 连续批处理 + serving `[新]` | ✅ |
| 44 | `44_inference_acceleration.ipynb` | KV cache、FlashAttention、投机解码、vLLM | ✅ |
| 45 | `45_distributed_training.ipynb` | DDP/FSDP/流水线并行/混合精度 | ✅ |
| 46 | `46_edge_deployment.ipynb` | 🔥WebGPU、本地部署 | ✅ |

### 板块八：对齐、安全与评估（5 notebooks）

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 47 | `47_alignment.ipynb` | Constitution AI、对齐范式(🪙机制设计) | ✅ |
| 48 | `48_safety_jailbreak.ipynb` | 红队、越狱、🔥abliteration | ✅ |
| 49 | `49_adversarial_robustness.ipynb` | FGSM/PGD/对抗训练 `[新2]` | ✅ |
| 50 | `50_ood_calibration.ipynb` | OOD 检测、温度缩放、置信度校准 `[新2]` | ✅ |
| 51 | `51_evaluation.ipynb` | benchmark 设计、生产评估 | ✅ |

### 板块九：Agent 与系统（9 notebooks）🧭🔥

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 52 | `52_agent_foundations.ipynb` | ReAct、Plan-Execute、ToT/GoT | ✅ |
| 53 | `53_tool_use_mcp.ipynb` | Function Calling + 🔥MCP 协议 | ✅ |
| 54 | `54_context_engineering.ipynb` | 🔥Context Loop/Harness、context 压缩/路由 | ✅ |
| 55 | `55_rag_agentic.ipynb` | RAG、Agentic RAG、GraphRAG | ✅ |
| 56 | `56_memory_systems.ipynb` | 记忆系统、MemGPT/Letta、长期记忆 | ✅ |
| 57 | `57_multi_agent_frameworks.ipynb` | 多agent、LangGraph/AutoGen/CrewAI（🐝） | ✅ |
| 58 | `58_coding_agents.ipynb` | 代码 Agent、SWE-bench | ✅ |
| 59 | `59_computer_use.ipynb` | Computer use/Browser use `[新]` | ✅ |
| 60 | `60_agent_evaluation.ipynb` | Agent 评估、benchmark | ✅ |

### 板块十：时序深度学习（2 notebooks）

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 61 | `61_time_series_models.ipynb` | PatchTST/iTransformer 时序模型 | ✅ |
| 62 | `62_time_series_foundation.ipynb` | TimesFM/Chronos/Moirai 时序基础模型 `[新]` | ✅ |

### 板块十一：AI for Science & 自动化发现（6 notebooks）

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 63 | `63_alphafold.ipynb` | AlphaFold、蛋白质结构预测 | ✅ |
| 64 | `64_protein_design.ipynb` | RFdiffusion、蛋白质设计 `[新]` | ✅ |
| 65 | `65_molecule_pinn.ipynb` | 分子生成、PINN/神经算子 | ✅ |
| 66 | `66_materials_discovery.ipynb` | GNoME 材料发现 `[新]` | ✅ |
| 67 | `67_weather_climate.ipynb` | GraphCast/Pangu 天气预测 `[新]` | ✅ |
| 68 | `68_automated_discovery.ipynb` | AlphaProof、🔥AlphaEvolve(LLM+进化) | ✅ |

### 板块十二：机制可解释性（3 notebooks）

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 69 | `69_sae_interpretability.ipynb` | 稀疏自编码器(SAE)、机制可解释性 `[新]` | ✅ |
| 70 | `70_representation_engineering.ipynb` | RepE/激活工程 `[新2]` | ✅ |
| 71 | `71_probing_attribution.ipynb` | 探针、归因、不确定性量化 | ✅ |

### 拓展章（2 notebooks）

| # | Notebook | 内容 | 状态 |
|---|----------|------|------|
| 72 | `72_deep_rl_frontiers.ipynb` | 🧭AlphaGo/AlphaZero历史、多智能体RL（🪙博弈论） | ✅ |
| 73 | `73_gnn.ipynb` | 图神经网络、Graph Transformer | ✅ |

---

## 五、进度追踪

### 总体进度

| 指标 | 数值 |
|------|------|
| 总 notebook 数 | 73 |
| 已完成 | 73 |
| 进行中 | 0 |
| 未开始 | 0 |
| 完成率 | 100% |

### 各板块进度

| 板块 | 总数 | 已完成 | 进度 |
|------|------|--------|------|
| 一、基础与范式演进 | 8 | 8 | 100% |
| 二、现代训练范式 | 10 | 10 | 100% |
| 三、推理与推理模型 | 4 | 4 | 100% |
| 四、多模态与具身 | 7 | 7 | 100% |
| 五、表征学习与检索 | 3 | 3 | 100% |
| 六、生成模型 | 8 | 8 | 100% |
| 七、效率与部署 | 6 | 6 | 100% |
| 八、对齐、安全与评估 | 5 | 5 | 100% |
| 九、Agent 与系统 | 9 | 9 | 100% |
| 十、时序深度学习 | 2 | 2 | 100% |
| 十一、AI for Science & 自动化发现 | 6 | 6 | 100% |
| 十二、机制可解释性 | 3 | 3 | 100% |
| 拓展章 | 2 | 2 | 100% |
| **合计** | **73** | **73** | **100%** |

---

## 六、章节依赖关系

### 前置依赖图（关键路径）

```
01 MLP+反向传播
  └→ 02 自动微分+优化器
       └→ 03 归一化+激活
            └→ 04 位置编码
                 ├→ 06 CNN↔ViT
                 └→ 07 RNN↔Transformer
                      └→ 08 架构创新(MoE/Mamba/GQA)
                           └→ 09 Tokenization
                                └→ 10 预训练+Scaling law
                                     ├→ 11 SFT+PEFT
                                     │    └→ 12 RLHF+PPO (🧭)
                                     │         └→ 13 DPO系列
                                     ├→ 19 推理模型
                                     │    └→ 20 PRM → 21 推理时方法
                                     └→ 23 CLIP/BLIP
                                          └→ 24 VLM → 25 世界模型
```

### 依赖说明

- **基础链**：01→02→03→04 是所有内容的基石，必须最先完成
- **Transformer 链**：07→08→09→10 是现代 LLM 的主干
- **RL 贯穿**：12(RLHF) 是 🧭 标记的起点，27(具身)/57(多agent)/72(深度RL) 依赖 RL 基础
- **博弈论就地讲**：39(GAN)/47(对齐)/57(多agent)/72(深度RL) 讲到时引入博弈论概念
- **生成模型链**：33(DDPM)→34(SD)→36(Flow Matching)→37(一致性模型) 有依赖
- **Agent 链**：52→53→54→55 有依赖，56/57/58/59 可并行

---

## 七、技术约定

### Notebook 结构约定
每个 notebook 建议包含以下部分：
1. **概述**：本章讲什么、为什么重要
2. **原理讲解**：算法/模型的数学原理，配推导
3. **代码实现**：PyTorch 实现，可运行
4. **可视化**：训练过程、中间结果、对比图
5. **实验**：在小数据集上跑通
6. **小结与延伸**：总结 + 指向下一章/相关章节

### 文件组织约定
- notebook 放在 `notebooks/` 目录下
- 配套数据放 `data/`
- 工具函数放 `utils/`
- 文件名使用 snake_case，编号两位数前缀

### 编码约定
- Python 代码遵循 PEP 8
- 文件名 snake_case
- UTF-8 编码
- 相对导入

---

## 八、变更日志

| 日期 | 变更 |
|------|------|
| 2026-09-03 | 初始框架定稿，73 notebooks，12 板块 + 拓展 |
| 2026-09-05 | 完成 `01_mlp_backprop.ipynb`：MLP+反向传播+计算图，14 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `02_autograd_optimizers.ipynb`：迷你autograd引擎+优化器演进+正则化，11 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `03_normalization_activation.ipynb`：归一化+激活函数演进+初始化，7 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `04_positional_encoding.ipynb`：位置编码演进(绝对→RoPE→ALiBi)，8 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `05_generalization_theory.ipynb`：泛化理论+双下降+万能近似定理，7 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `06_cnn_vs_vit.ipynb`：CNN↔ViT对比(归纳偏置vs数据效率)，7 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `07_rnn_vs_transformer.ipynb`：RNN/LSTM↔Transformer对比(长距离依赖+并行性)，12 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `08_architecture_innovations.ipynb`：MoE/SSM-Mamba/线性注意力/GQA架构创新，10 个 code cell 全部验证通过。**板块一(基础与范式演进)100%完成** |
| 2026-09-05 | 完成 `09_tokenization.ipynb`：BPE/WordPiece/Unigram从零实现+tiktoken，9 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `10_pretraining_scaling.ipynb`：CLM/MLM/MAE预训练+Scaling Law+Chinchilla定律，8 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `11_sft_peft.ipynb`：SFT+LoRA/QLoRA/DoRA参数高效微调，10 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `12_rlhf_ppo.ipynb`：🧭RLHF三阶段+奖励模型+PPO+KL惩罚完整实现，8 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `13_dpo_family.ipynb`：🔥DPO/GRPO/DAPO/GSPO直接偏好优化，6 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `14_data_engineering.ipynb`：合成数据+质量筛选+数据配比+自我改进，7 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `15_training_techniques.ipynb`：warmup+cosine调度+梯度累积+裁剪+混合精度，8 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `16_model_merging.ipynb`：SLERP/TIES/DARE模型合并，6 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `17_retrieval_augmented_training.ipynb`：kNN-LM/RETRO检索增强训练，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `18_non_gradient_optimization.ipynb`：ES/神经进化/DARTS非梯度优化，6 个 code cell 全部验证通过。**板块二(现代训练范式)100%完成** |
| 2026-09-05 | 完成 `19_reasoning_models.ipynb`：o1/R1/test-time scaling推理模型，6 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `20_prm_process_reward.ipynb`：PRM过程奖励模型+MCTS自举，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `21_inference_time_methods.ipynb`：best-of-N/self-consistency/verification推理时方法，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `22_long_context.ipynb`：Ring Attention+位置编码外推(NTK/YaRN)+KV cache压缩，4 个 code cell 全部验证通过。**板块三(推理与推理模型)100%完成** |
| 2026-09-05 | 完成 `23_clip_blip.ipynb`：CLIP双塔对比学习+InfoNCE+零样本分类+BLIP自举，6 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `24_vlm.ipynb`：BLIP-2 Q-Former+LLaVA线性投影+原生多模态+VLM评估，6 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `25_world_models_video.ipynb`：DiT+时空patch+Sora架构+世界模型，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `26_3d_representation.ipynb`：NeRF体渲染+3D Gaussian Splatting+位置编码，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `27_embodied_ai.ipynb`：RT-1/RT-2 VLA+动作文本化+BC vs RL+Sim-to-Real，6 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `28_embodied_simulation.ipynb`：Isaac Gym并行环境+向量化PPO+Habitat导航+仿真器演进，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `29_speech_audio.ipynb`：语音特征提取+CTC+Whisper多任务+TTS演进+统一语音模型，6 个 code cell 全部验证通过。**板块四(多模态与具身)100%完成** |
| 2026-09-05 | 完成 `30_contrastive_learning.ipynb`：SimCLR+MoCo队列+InfoNCE+温度参数，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `31_non_contrastive_ssl.ipynb`：BYOL无负样本+DINO自蒸馏+DINOv2超监督+三大范式，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `32_embedding_retrieval.ipynb`：Flat/IVF/HNSW/PQ向量检索+10B规模+ColBERT多向量，6 个 code cell 全部验证通过。**板块五(表征学习与检索)100%完成** |
| 2026-09-05 | 完成 `33_ddpm.ipynb`：DDPM前向/反向扩散+DDIM加速采样+瑞士卷生成，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `34_stable_diffusion.ipynb`：Latent Diffusion+cross-attention+CFG+完整SD流程，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `35_controlnet.ipynb`：ControlNet零卷积+多控制组合+IP-Adapter参考图注入，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `36_flow_matching.ipynb`：Flow Matching连续流+Rectified Flow拉直路径+与DDPM对比，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `37_consistency_models.ipynb`：一致性模型+一步生成+蒸馏vs训练，2 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `38_dit.ipynb`：DiT架构+adaLN-Zero条件注入+scaling law，3 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `39_vae_gan_flow.ipynb`：VAE变分下界+GAN纳什均衡+Normalizing Flow可逆变换，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `40_autoregressive_vs_diffusion.ipynb`：自回归vs扩散对比+融合趋势+各领域适用性，4 个 code cell 全部验证通过。**板块六(生成模型)100%完成** |
| 2026-09-05 | 完成 `41_distillation_quantization.ipynb`：知识蒸馏+PTQ/QAT量化+GPTQ/AWQ+4bit量化，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `42_pruning_moe.ipynb`：幅度剪枝+稀疏训练+MoE路由+专家容量，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `43_paged_attention_serving.ipynb`：PagedAttention分页KV+连续批处理+vLLM serving架构，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `44_inference_acceleration.ipynb`：KV cache+FlashAttention+投机解码+推理加速对比，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `45_distributed_training.ipynb`：DDP/FSDP+3D并行(数据/张量/流水线)+混合精度+Ring Attention，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `46_edge_deployment.ipynb`：ONNX导出+INT8量化+WebGPU部署+边缘设备对比，4 个 code cell 全部验证通过。**板块七(效率与部署)100%完成** |
| 2026-09-05 | 完成 `47_alignment.ipynb`：对齐HHH原则+范式演进(RLHF→DPO→CAI→RLAIF)+Constitutional AI自我批评+机制设计视角+奖励黑客，7 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `48_safety_jailbreak.ipynb`：红队测试+越狱技术分类+GCG自动越狱+abliteration消融对齐+安全护栏多层防御+攻防博弈，7 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `49_adversarial_robustness.ipynb`：对抗样本+FGSM/PGD攻击+对抗训练+认证防御(随机平滑)+鲁棒性vs准确率权衡，7 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `50_ood_calibration.ipynb`：置信度校准+可靠性图+ECE+温度缩放+OOD检测(MSP/能量/马氏)+MC Dropout不确定性量化，8 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `51_evaluation.ipynb`：评估挑战+主要benchmark(MMLU/HumanEval/GSM8K)+设计原则+数据污染+LLM-as-Judge偏差+生产A/B测试+评估漏斗，8 个 code cell 全部验证通过。**板块八(对齐、安全与评估)100%完成** |
| 2026-09-05 | 完成 `52_agent_foundations.ipynb`：Agent定义+ReAct(推理+行动)+Plan-Execute+ToT思维树搜索+GoT思维图+范式对比，7 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `53_tool_use_mcp.ipynb`：Function Calling机制+MCP协议(标准化)+工具设计+工具编排+并行调用，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `54_context_engineering.ipynb`：上下文工程+Context Loop+压缩(摘要/选择性)+路由(相关性检索)+中间迷失+窗口管理策略，6 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `55_rag_agentic.ipynb`：Naive RAG+Advanced RAG(改写+重排)+Agentic RAG(迭代检索)+GraphRAG(知识图谱+社区)+范式对比，6 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `56_memory_systems.ipynb`：三层记忆(短期/长期/反思)+MemGPT/Letta OS思想+记忆操作(写入/检索/遗忘)+策略对比，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `57_multi_agent_frameworks.ipynb`：多Agent协作模式+LangGraph(图结构)+AutoGen(对话式)+CrewAI(角色分工)+框架对比，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `58_coding_agents.ipynb`：代码Agent演进+SWE-bench评估+Agent架构(理解→规划→编辑→测试)+模型规模对比，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `59_computer_use.ipynb`：Computer Use(操作电脑)+Browser Use(浏览器自动化)+动作空间+视觉理解+应用场景，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `60_agent_evaluation.ipynb`：Agent评估挑战+多维度评估(成功率/效率/成本/安全)+主要benchmark+分步评估+最佳实践，5 个 code cell 全部验证通过。**板块九(Agent与系统)100%完成** |
| 2026-09-05 | 完成 `61_time_series_models.ipynb`：时序数据特性+PatchTST(patch+Transformer)+iTransformer(变量反转)+范式对比，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `62_time_series_foundation.ipynb`：时序基础模型(TimesFM/Chronos/Moirai)+零样本预测+预训练策略+scaling law，5 个 code cell 全部验证通过。**板块十(时序深度学习)100%完成** |
| 2026-09-05 | 完成 `63_alphafold.ipynb`：AlphaFold架构(Evoformer+结构模块)+MSA+不变点注意力(AF2)+AF3扩散，5 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `64_protein_design.ipynb`：蛋白质设计(反向折叠)+RFdiffusion(扩散生成)+MPNN+对称性约束+应用，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `65_molecule_pinn.ipynb`：分子图表示+PINN物理约束+神经算子(FNO/DeepONet)+科学计算应用，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `66_materials_discovery.ipynb`：GNoME材料发现+图网络筛选+凸包稳定性+对称性+自动化实验闭环，3 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `67_weather_climate.ipynb`：GraphCast(GNN全球天气)+Pangu(Transformer)+物理约束+AI预报vs传统NWP+气候建模，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `68_automated_discovery.ipynb`：AlphaProof(形式化证明)+AlphaEvolve(LLM+进化算法)+自动发现闭环+科学发现范式，3 个 code cell 全部验证通过。**板块十一(AI for Science)100%完成** |
| 2026-09-05 | 完成 `69_sae_interpretability.ipynb`：稀疏自编码器(SAE)+特征字典+机制可解释性+单义性+超完备基+训练技巧，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `70_representation_engineering.ipynb`：RepE表征工程+激活向量+概念向量+ steering+对比模型行为+RepE vs SAE，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `71_probing_attribution.ipynb`：探针(线性/非线性)+归因(梯度×输入/积分梯度/注意力)+不确定性量化+可解释性方法对比，6 个 code cell 全部验证通过。**板块十二(机制可解释性)100%完成** |
| 2026-09-05 | 完成 `72_deep_rl_frontiers.ipynb`：🧭AlphaGo/AlphaZero(自我对弈+MCTS)+多智能体RL(协作/竞争)+🪙博弈论(纳什均衡)+OpenAI Five/AlphaStar，4 个 code cell 全部验证通过 |
| 2026-09-05 | 完成 `73_gnn.ipynb`：图数据基础(邻接矩阵/度矩阵)+消息传递范式+GCN图卷积+GAT图注意力+Graph Transformer+GNN对比，6 个 code cell 全部验证通过。**拓展章100%完成。全部73个notebook完成 🎉** |

---

> **使用方式**：实现每个 notebook 时，将对应行的状态从 ⬜ 改为 🔄，完成后改为 ✅，并更新第五节的进度表。