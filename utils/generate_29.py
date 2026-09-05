# 生成 29_speech_audio.ipynb 的脚本
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}
def md(text): nb.cells.append(nbf.v4.new_markdown_cell(text))
def code(text): nb.cells.append(nbf.v4.new_code_cell(text))

md("""# 29 — 语音与音频：ASR、TTS、Whisper

> 语音是人类最自然的交互方式。Whisper 让 ASR 接近人类水平，
> VALL-E/Tortoise 让 TTS 能克隆音色。

## 本章你将掌握

1. **语音处理基础**：波形、频谱、梅尔频率
2. **ASR 演进**：HMM → CTC → Attention → Whisper
3. **TTS 演进**：拼接 → 参数 → 神经 → VALL-E
4. **Whisper 架构**：多语言、多任务""")

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

md("""## 1. 语音处理基础

### 1.1 声音是什么

```
声音 = 空气压力的周期性变化
数字化: 以采样率 fs 采样 → 一维数组

常见采样率:
  电话: 8 kHz
  音频 CD: 44.1 kHz
  语音识别: 16 kHz
```

### 1.2 从波形到频谱

直接用波形很难学习。通常转换到**频域**：

```
STFT (短时傅里叶变换):
  波形 → 窗口化 → FFT → 频谱图 [时间, 频率]

梅尔频率:
  人耳对低频敏感, 对高频不敏感
  梅尔刻度模拟人耳感知
  → mel-spectrogram 是语音处理的标配
```

### 1.3 特征提取流程

```
波形 (16000 Hz)
  → 预加重 (高通滤波)
  → 分帧 (25ms 窗口, 10ms 步长)
  → 加窗 (汉明窗)
  → FFT → 功率谱
  → 梅尔滤波器组 → mel-spectrogram
  → log → log-mel-spectrogram
```

> 💡 log-mel-spectrogram 是几乎所有现代语音模型的输入。
> 它模拟人耳的频率感知，同时对数压缩动态范围。""")

code("""# 语音特征提取模拟
def generate_speech_like_signal(duration=2.0, sr=16000):
    t = np.linspace(0, duration, int(sr * duration))
    # 模拟语音: 基频 + 谐波 + 噪声
    f0 = 150  # 基频 150Hz (男声)
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    # 加噪声
    signal += 0.1 * np.random.randn(len(t))
    # 调制 (模拟音节)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    signal *= envelope
    return t, signal, sr

def simple_stft(signal, n_fft=512, hop=160, n_mels=80):
    # 简化 STFT
    n_frames = (len(signal) - n_fft) // hop + 1
    spectrogram = []
    for i in range(n_frames):
        frame = signal[i * hop : i * hop + n_fft]
        window = np.hamming(n_fft)
        fft = np.fft.rfft(frame * window)
        power = np.abs(fft) ** 2
        spectrogram.append(power[:n_mels])
    return np.array(spectrogram).T

t, signal, sr = generate_speech_like_signal()
spectrogram = simple_stft(signal)
log_mel = np.log(spectrogram + 1e-10)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(t[:1000], signal[:1000], linewidth=0.5)
axes[0].set_xlabel('时间 (s)'); axes[0].set_ylabel('振幅')
axes[0].set_title('语音波形 (前 0.06s)')

im = axes[1].imshow(log_mel, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_xlabel('帧'); axes[1].set_ylabel('频率 bin')
axes[1].set_title('Log-Mel 频谱图')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig('notebooks/fig_speech_features.png', bbox_inches='tight')
plt.show()
print(f"波形: {len(signal)} 采样点 ({len(signal)/sr:.1f}s @ {sr}Hz)")
print(f"频谱图: {spectrogram.shape} (频率 × 时间)")
print("Log-Mel 频谱图: 语音模型的标配输入。")""")

md("""## 2. ASR 演进：从 HMM 到 Whisper

### 2.1 四个时代

```
1. HMM 时代 (2000s): GMM-HMM → DNN-HMM
   特征: MFCC + 隐马尔可夫模型
   缺点: 需要大量手工设计

2. CTC 时代 (2010s): Connectionist Temporal Classification
   特征: 端到端, 允许重复/blank token
   代表: DeepSpeech, wav2letter

3. Attention 时代 (2018+): Seq2Seq + Attention
   特征: 编码器-解码器, 注意力对齐
   代表: Listen Attend Spell, Conformer

4. Whisper 时代 (2022+): 多任务, 多语言
   特征: 弱监督, 68万小时多语言数据
   代表: Whisper
```

### 2.2 CTC 损失

CTC 解决**对齐问题**：语音长度 ≠ 文字长度，如何对齐？

```
语音: 100 帧
文字: "hello" (5 字符)

CTC 允许: h-h-e-e-l-l-l-l-o-o → 合并重复 → "hello"
          blank-blank-h-... → 去除 blank → "hello"

对所有可能对齐求和 → CTC 损失
```

> 💡 CTC 的天才：用 blank token 处理重复，自动学习对齐。
> 但 CTC 假设帧间独立，所以后来加上了语言模型。""")

code("""# CTC 简化演示
def ctc_decode(predictions, blank_id=0):
    # 简化 CTC 解码: 去除 blank 和重复
    result = []
    prev = None
    for p in predictions:
        if p != blank_id and p != prev:
            result.append(p)
        prev = p
    return result

# 模拟 CTC 输出: 每帧预测一个 token
# 0=blank, 1=h, 2=e, 3=l, 4=o
np.random.seed(42)
frame_predictions = [0, 0, 1, 1, 0, 2, 2, 0, 3, 3, 3, 0, 3, 0, 4, 4, 0]
decoded = ctc_decode(frame_predictions)

token_map = {1: 'h', 2: 'e', 3: 'l', 4: 'o'}
decoded_text = ''.join(token_map[t] for t in decoded)

print("CTC 解码演示:")
print(f"  帧预测: {frame_predictions}")
print(f"  解码: {decoded} → '{decoded_text}'")
print(f"  (0=blank, 去除重复和blank)")

# 可视化
fig, ax = plt.subplots(figsize=(12, 3))
colors = ['gray', 'red', 'green', 'blue', 'orange']
labels = ['blank', 'h', 'e', 'l', 'o']
for i, p in enumerate(frame_predictions):
    ax.bar(i, 1, color=colors[p], edgecolor='black')
    ax.text(i, 0.5, labels[p], ha='center', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('帧'); ax.set_title(f'CTC 解码: 帧序列 → "{decoded_text}"')
ax.set_xlim(-0.5, len(frame_predictions) - 0.5)
ax.set_yticks([])
plt.tight_layout()
plt.savefig('notebooks/fig_ctc.png', bbox_inches='tight')
plt.show()
print("CTC: 自动学习帧到字符的对齐。")""")

md("""## 3. Whisper：多任务多语言 ASR

### 3.1 核心创新

Whisper（OpenAI 2022）用 **68 万小时**多语言弱监督数据训练：

```
数据: 从互联网爬取 (音频, 字幕) 对
  - 弱监督: 字幕可能有误
  - 多语言: 99 种语言
  - 多任务: 转写 + 翻译 + 语言识别 + 时间戳

架构: Encoder-Decoder Transformer
  音频 → mel → Encoder → 隐藏状态
  隐藏状态 + <前文> → Decoder → 下一个 token
```

### 3.2 多任务能力

```
输入: [音频]
输出: [<|zh|> <|transcribe|> 你好世界 <|endoftext|>]

特殊 token 控制任务:
  <|zh|>         → 中文转写
  <|en|>         → 英文转写
  <|translate|>  → 翻译到英文
  <|notimestamps|> → 不要时间戳
```

### 3.3 为什么 Whisper 好

- **多语言**：99 种语言，低资源语言也能用
- **多任务**：转写 + 翻译 + 检测
- **鲁棒**：噪声、口音、不同质量音频
- **零样本**：不需要微调

> 💡 Whisper 的成功靠的是**数据规模**而非架构创新。
# 68 万小时弱监督数据 > 精标的少量数据。""")

code("""# Whisper 架构简化
class MiniWhisper(nn.Module):
    def __init__(self, d_model=256, n_mels=80, vocab_size=1000):
        super().__init__()
        # 音频编码器
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(n_mels, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*4, batch_first=True),
            num_layers=4
        )
        # 文本解码器
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=4, dim_feedforward=d_model*4, batch_first=True),
            num_layers=4
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, mel_spec, text_tokens):
        # 编码音频
        audio_feat = self.audio_encoder(mel_spec.transpose(1, 2)).transpose(1, 2)
        audio_hidden = self.encoder(audio_feat)

        # 解码文本
        text_embed = self.text_embed(text_tokens)
        decoded = self.decoder(text_embed, audio_hidden)
        logits = self.head(decoded)
        return logits

# 演示 Whisper
whisper = MiniWhisper(d_model=256, n_mels=80, vocab_size=1000)
mel_spec = torch.randn(2, 100, 80)  # 2条音频, 100帧, 80 mel
text_tokens = torch.randint(0, 1000, (2, 10))  # 10个token

logits = whisper(mel_spec, text_tokens)
print(f"输入: mel {mel_spec.shape} + 文本 {text_tokens.shape}")
print(f"输出: logits {logits.shape}")
print(f"参数量: {sum(p.numel() for p in whisper.parameters())/1e6:.1f}M")
print("Whisper: Encoder-Decoder, 多任务多语言。")""")

md("""## 4. TTS 演进：从拼接到生成

### 4.1 四个时代

```
1. 拼接 TTS (1990s): 从录音库拼接音素
   优点: 自然; 缺点: 需要大量录音, 不能泛化

2. 参数 TTS (2000s): HMM/参数模型生成语音参数
   优点: 灵活; 缺点: 听起来机械

3. 神经 TTS (2017+): Tacotron, WaveNet, FastSpeech
   优点: 自然; 缺点: 仍需大量录音

4. 生成式 TTS (2023+): VALL-E, Tortoise, NaturalSpeech
   优点: 少样本音色克隆; 缺点: 计算量大
```

### 4.2 神经 TTS 流程

```
文本 → 文本前端 → 音素序列
  → 声学模型 → mel-spectrogram
  → 声码器 (vocoder) → 波形
```

### 4.3 VALL-E：少样本音色克隆

VALL-E 用 **3 秒**录音就能克隆音色：

```
输入: 3秒目标音色录音 + 要说的文本
输出: 目标音色说新文本

原理: 把语音离散化 token, 用语言模型生成
  类似 DALL-E 对图像, VALL-E 对语音
```

> 💡 VALL-E 的哲学：语音也是一种"语言"，用语言模型生成。
# 3 秒克隆音色——以前需要几小时录音。""")

code("""# TTS 流程模拟
def tts_pipeline_demo(text, speaker_id=0):
    # 1. 文本前端: 文本 → 音素
    phonemes = list(text)  # 简化: 每个字一个音素
    n_phonemes = len(phonemes)

    # 2. 声学模型: 音素 → mel-spectrogram
    np.random.seed(speaker_id)
    mel = np.random.randn(80, n_phonemes * 10) * 0.5 + 2  # [80, T]

    # 3. 声码器: mel → 波形
    waveform = np.random.randn(n_phonemes * 10 * 160) * 0.1  # 简化

    return phonemes, mel, waveform

# 不同说话人的 TTS
texts = ["你好世界", "语音合成"]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for i, text in enumerate(texts):
    for j, speaker in enumerate(['说话人A', '说话人B']):
        phonemes, mel, waveform = tts_pipeline_demo(text, speaker_id=j)
        ax = axes[i, j]
        ax.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'"{text}" - {speaker}')
        ax.set_xlabel('帧'); ax.set_ylabel('Mel bin')

plt.tight_layout()
plt.savefig('notebooks/fig_tts.png', bbox_inches='tight')
plt.show()
print("TTS: 文本 → mel频谱 → 波形")
print("VALL-E: 3秒录音克隆音色——语音作为语言。")""")

md("""## 5. 语音模型的前沿

### 5.1 音频大模型

```
AudioLM:  音频 token 生成 (类似 LLM)
MusicGen: 文本 → 音乐
VALL-E:   少样本 TTS
Whisper:  多语言 ASR
Pengi:    音频理解 + 问答
```

### 5.2 统一语音模型

趋势：一个模型做所有语音任务：

```
输入: 音频 + 任务描述
输出: 文本/音频

任务:
  - ASR: 转写
  - TTS: 合成
  - 翻译: 语音翻译
  - 问答: 语音问答
  - 生成: 音乐/音效生成
```

### 5.3 实时语音交互

GPT-4o 的语音模式：
- 端到端语音到语音（不经过文本中间表示）
- 实时响应（<300ms）
- 情感保留（笑声、犹豫）

> 💡 GPT-4o 的突破：**端到端语音到语音**，不经过 ASR→LLM→TTS 管道。
# 这保留了情感、语调，延迟也更低。""")

code("""# 语音任务统一模型概念
fig, ax = plt.subplots(figsize=(12, 6))

tasks = ['ASR\\n(语音→文本)', 'TTS\\n(文本→语音)', '翻译\\n(语音→语音)',
         '分类\\n(情感/语种)', '生成\\n(音乐/音效)', '问答\\n(语音问答)']
accuracies = [95, 90, 88, 92, 85, 80]

bars = ax.barh(tasks, accuracies, color='steelblue', edgecolor='navy', alpha=0.8)
ax.set_xlabel('能力 (%)')
ax.set_title('统一语音模型: 一个模型做所有任务')
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{acc}%', va='center', fontsize=11)
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig('notebooks/fig_unified_speech.png', bbox_inches='tight')
plt.show()
print("趋势: 一个模型做所有语音任务——类似 LLM 在文本上的统一。")""")

md("""## 6. 小结

### ✅ 本章你掌握了
| 概念 | 状态 |
|------|------|
| 语音特征提取 | ✅ |
| ASR 演进 (HMM→CTC→Whisper) | ✅ |
| CTC 损失 | ✅ |
| Whisper 多任务 | ✅ |
| TTS 演进 | ✅ |
| 统一语音模型 | ✅ |

### 核心 takeaway
> **Whisper 靠数据规模取胜，VALL-E 把语音当语言**。
> 从 ASR/TTS 分离到 GPT-4o 的端到端语音——语音正在被统一。
> 语音是最自然的交互方式，端到端语音模型是未来。

### 🔗 下一板块
**`30_representation_learning.ipynb`** — 表征学习、对比学习、SimCLR（进入板块五）

---

> 💬 **板块四(多模态与具身)完结。**""")

output_path = "notebooks/29_speech_audio.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ {output_path} ({len(nb.cells)} cells)")