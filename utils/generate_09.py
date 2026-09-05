"""生成 09_tokenization.ipynb 的脚本"""
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
md("""# 09 — Tokenization：从字符到子词

> LLM 不直接读文本——它读 **token 序列**。怎么把文本切成 token，这看似简单的问题
> 却深刻影响模型的词表大小、OOV 处理、多语言能力。现代 LLM 几乎都用 **BPE** 或其变体。
>
> 本章从零实现 BPE/WordPiece/Unigram 三大算法，对比它们的特性。

## 本章你将掌握

1. **字符级 vs 词级**：为什么都不行
2. **BPE**：字节对编码，GPT 系列用的
3. **WordPiece**：BERT 用的，类似 BPE 但用似然选择
4. **Unigram**：SentencePiece 的一种，概率删除
5. **tiktoken**：GPT-4 的 BPE 实现""")

code("""import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.random.seed(42)
print("环境准备完成 ✅")""")

# ============================================================
md("""## 1. 为什么需要 Tokenization

### 1.1 两种极端方案

| 方案 | 词表 | 优点 | 缺点 |
|------|------|------|------|
| **字符级** | ~100 | 词表小，无 OOV | 序列太长，语义稀疏 |
| **词级** | ~100K+ | 语义完整，序列短 | OOV 严重，词表巨大 |

```
字符级: "hello world" → ['h','e','l','l','o',' ','w','o','r','l','d']  (11个token)
词  级: "hello world" → ['hello', 'world']                              (2个token)
子词级: "hello world" → ['hello', ' world']                             (2个token, 但能处理新词)
```

### 1.2 子词：两全其美

**子词（subword）** 在字符和词之间找平衡：
- 高频词保持完整（如 "hello"）
- 低频词拆成子词（如 "tokenization" → "token" + "ization"）
- 永远没有 OOV——最差拆成单字符

> 💡 这就是为什么 GPT 用 BPE——"unbelievable" 可能不在词表，但 "un" + "believ" + "able" 在。""")

code("""# 演示字符级 vs 词级的问题
text = "The tokenization is unbelievably complicated."

# 字符级
char_tokens = list(text)
print(f"字符级: {len(char_tokens)} 个 token")
print(f"  {char_tokens[:15]}...")

# 词级
word_tokens = text.split()
print(f"\\n词级: {len(word_tokens)} 个 token")
print(f"  {word_tokens}")

# 词级的问题: 新词 OOV
new_text = "The chatgptification is unprecedented."
new_words = new_text.split()
print(f"\\n新文本词级: {new_words}")
print("  'chatgptification' 不在词表 → OOV! 'unprecedented' 也可能不在。")
print("  子词方案: 'chat' + 'gpt' + 'ification', 'un' + 'precedent' + 'ed' → 无 OOV")""")

# ============================================================
md("""## 2. BPE：字节对编码

### 2.1 BPE 算法

BPE 的思想极其简单——**反复合并最常见的相邻对**：

```
初始: 每个字符是一个 token
循环:
  1. 统计所有相邻 token 对的频率
  2. 找到频率最高的对 (a, b)
  3. 把所有 (a, b) 合并成新 token "ab"
  4. 加入词表
直到: 词表达到目标大小
```

### 2.2 示例

```
语料: "low low low low low lower lower newest newest newest newest newest newest wider wider new new"
初始: l o w   l o w   ...
对频率: (l,o)=22, (o,w)=22, (e,r)=2, (n,e)=6, (w,e)=8...
合并 (l,o)→lo: lo w   lo w   ...
合并 (lo,w)→low: low   low   ...
...""")

code("""class SimpleBPE:
    \"\"\"从零实现 BPE (Byte Pair Encoding)\"\"\"
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.merges = []  # 合并规则列表
        self.vocab = set()

    def train(self, texts):
        \"\"\"在语料上训练 BPE\"\"\"
        # 初始: 每个词拆成字符 + 结束符
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                # 用空格分隔字符, 加 </w> 标记词尾
                char_seq = ' '.join(list(word)) + ' </w>'
                word_freqs[char_seq] += 1

        # 逐步合并
        for _ in range(self.num_merges):
            # 统计相邻对频率
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i+1])] += freq

            if not pair_freqs:
                break

            # 找最高频对
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges.append(best_pair)

            # 执行合并
            new_word_freqs = Counter()
            bigram = ' '.join(best_pair)
            replacement = ''.join(best_pair)
            for word, freq in word_freqs.items():
                new_word = word.replace(bigram, replacement)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs

        # 构建词表
        self.vocab = set()
        for word in word_freqs:
            self.vocab.update(word.split())

    def encode(self, word):
        \"\"\"编码单个词\"\"\"
        symbols = list(word) + ['</w>']
        for a, b in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i+1] == b:
                    symbols = symbols[:i] + [a+b] + symbols[i+2:]
                else:
                    i += 1
        return symbols

# 训练 BPE
corpus = [
    "low low low low low lower lower newest newest newest newest newest newest wider wider new new",
    "newer newer newer newer lower lower lowest lowest widest widest"
]
bpe = SimpleBPE(num_merges=15)
bpe.train(corpus)

print("BPE 合并规则 (前10个):")
for i, (a, b) in enumerate(bpe.merges[:10]):
    print(f"  {i+1:2d}. '{a}' + '{b}' → '{a+b}'")

print(f"\\n词表大小: {len(bpe.vocab)}")
print(f"词表: {sorted(bpe.vocab)[:15]}...")""")

code("""# 编码新词
test_words = ["low", "lower", "lowest", "newest", "newer", "wider", "widest", "unknown"]
print("BPE 编码结果:")
for word in test_words:
    tokens = bpe.encode(word)
    print(f"  {word:12s} → {tokens}")

print("\\n注意: 即使是训练时没见过的 'unknown', 也能拆成子词——永远没有 OOV!")""")

# ============================================================
md("""## 3. WordPiece：BERT 用的

### 3.1 WordPiece vs BPE

WordPiece 和 BPE 很像，但选择合并对的准则不同：

| | BPE | WordPiece |
|---|------|-----------|
| **选择准则** | 最高频率 | 最高似然增益 |
| **公式** | $\\max \\text{freq}(a,b)$ | $\\max \\frac{\\text{freq}(a,b)}{\\text{freq}(a) \\cdot \\text{freq}(b)}$ |
| **词表标记** | `</w>` 词尾 | `##` 词内续 |

WordPiece 的似然增益公式偏好**合并各自频率低的对**——避免高频词被过度合并。

### 3.2 BERT 的词表

BERT 用 WordPiece，词表大小 30K。词内续用 `##` 前缀：
```
"playing" → ["play", "##ing"]
"unbelievable" → ["un", "##bel", "##ie", "##able"]
```""")

code("""class SimpleWordPiece:
    \"\"\"简化 WordPiece: 用似然增益选择合并对\"\"\"
    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.merges = []

    def train(self, texts):
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                char_seq = ' '.join(list(word)) + ' ##end'
                word_freqs[char_seq] += 1

        for _ in range(self.num_merges):
            pair_freqs = Counter()
            single_freqs = Counter()
            for word, freq in word_freqs.items():
                symbols = word.split()
                for s in symbols:
                    single_freqs[s] += freq
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i+1])] += freq

            if not pair_freqs:
                break

            # WordPiece: 用似然增益 = freq(a,b) / (freq(a) * freq(b))
            def score(pair):
                a, b = pair
                return pair_freqs[pair] / (single_freqs[a] * single_freqs[b] + 1e-10)

            best_pair = max(pair_freqs, key=score)
            self.merges.append(best_pair)

            bigram = ' '.join(best_pair)
            replacement = ''.join(best_pair)
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = word.replace(bigram, replacement)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs

    def encode(self, word):
        symbols = list(word) + ['##end']
        for a, b in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i+1] == b:
                    symbols = symbols[:i] + [a+b] + symbols[i+2:]
                else:
                    i += 1
        return symbols

wp = SimpleWordPiece(num_merges=15)
wp.train(corpus)

print("WordPiece 合并规则 (前10个):")
for i, (a, b) in enumerate(wp.merges[:10]):
    print(f"  {i+1:2d}. '{a}' + '{b}' → '{a+b}'")

print("\\n对比 BPE 和 WordPiece 的前5个合并:")
print(f"  BPE:       {bpe.merges[:5]}")
print(f"  WordPiece: {wp.merges[:5]}")
print("WordPiece 倾向于先合并低频字符对, BPE 倾向于先合并高频对。")""")

# ============================================================
md("""## 4. Unigram：SentencePiece 的另一种

### 4.1 Unigram 的思想

Unigram 反过来——**从一个大词表开始，逐步删除降低似然最少的子词**：

```
初始: 生成所有可能的子词候选 (大量)
循环:
  1. 计算每个子词对总似然的贡献
  2. 删除贡献最小的 (保留 80%)
  3. 重新计算语料的最佳分词
直到: 词表达到目标大小
```

### 4.2 Unigram vs BPE

| | BPE | Unigram |
|---|------|---------|
| **方向** | 自底向上（合并） | 自顶向下（删除） |
| **分词** | 确定性 | **概率性**（有多种分词方式） |
| **优势** | 简单快速 | 能选最优分词，更灵活 |

> 💡 Unigram 的概率分词是优势——"tokenization" 可以是 "token+ization" 或 "tok+enization"，
> Unigram 能根据概率选更好的。SentencePiece 同时支持 BPE 和 Unigram。""")

code("""class SimpleUnigram:
    \"\"\"简化 Unigram 模型\"\"\"
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab_probs = {}  # 子词 → 对数概率

    def _generate_candidates(self, texts, max_len=6):
        \"\"\"生成所有可能的子词候选\"\"\"
        candidates = Counter()
        for text in texts:
            for word in text.split():
                for i in range(len(word)):
                    for j in range(i+1, min(len(word)+1, i+max_len+1)):
                        candidates[word[i:j]] += 1
        return candidates

    def train(self, texts):
        # 生成候选
        candidates = self._generate_candidates(texts)
        # 初始词表: 所有候选, 概率按频率
        total = sum(candidates.values())
        self.vocab_probs = {sub: np.log(freq / total) for sub, freq in candidates.items()}

        # 逐步删除
        while len(self.vocab_probs) > self.vocab_size:
            # 计算每个子词的损失 (简化: 用频率近似)
            # 真实 Unigram 用 EM 算法计算每个子词的似然贡献
            # 这里简化: 删除频率最低的
            sorted_vocab = sorted(self.vocab_probs.items(), key=lambda x: x[1])
            # 删除最低的 20%
            n_delete = max(1, int(0.2 * len(sorted_vocab)))
            for sub, _ in sorted_vocab[:n_delete]:
                del self.vocab_probs[sub]
            if len(self.vocab_probs) <= self.vocab_size:
                break

    def encode(self, word):
        \"\"\"用动态规划找最优分词 (Viterbi)\"\"\"
        n = len(word)
        # dp[i] = (最佳分词, 最佳对数概率)
        dp = [(None, -float('inf')) for _ in range(n + 1)]
        dp[0] = ([], 0.0)

        for i in range(1, n + 1):
            for j in range(max(0, i - 6), i):  # 最多回看6个字符
                sub = word[j:i]
                if sub in self.vocab_probs:
                    if dp[j][1] + self.vocab_probs[sub] > dp[i][1]:
                        dp[i] = (dp[j][0] + [sub], dp[j][1] + self.vocab_probs[sub])
            # 如果没有子词匹配, 用单字符 (OOV fallback)
            if dp[i][0] is None:
                char = word[i-1:i]
                fallback_prob = self.vocab_probs.get(char, -10.0)
                dp[i] = (dp[i-1][0] + [char], dp[i-1][1] + fallback_prob)

        return dp[n][0]

unigram = SimpleUnigram(vocab_size=30)
unigram.train(corpus)

print(f"Unigram 词表大小: {len(unigram.vocab_probs)}")
print(f"词表 (前15): {sorted(unigram.vocab_probs.keys(), key=lambda x: -unigram.vocab_probs[x])[:15]}")

print("\\nUnigram 编码 (用 Viterbi 找最优分词):")
for word in ["low", "lower", "newest", "newer"]:
    tokens = unigram.encode(word)
    print(f"  {word:12s} → {tokens}")""")

# ============================================================
md("""## 5. 对比三种 Tokenizer""")

code("""# 对比编码结果
test_words = ["low", "lower", "lowest", "newest", "newer", "wider"]
print(f"{'词':12s} | {'BPE':30s} | {'WordPiece':30s} | {'Unigram':30s}")
print("-" * 110)
for word in test_words:
    bpe_tokens = bpe.encode(word)
    wp_tokens = wp.encode(word)
    uni_tokens = unigram.encode(word)
    print(f"{word:12s} | {str(bpe_tokens):30s} | {str(wp_tokens):30s} | {str(uni_tokens):30s}")

print("\\n三种 tokenizer 都能把新词拆成子词——永远没有 OOV。")
print("但拆法不同: BPE 偏频率, WordPiece 偏似然, Unigram 偏概率最优。")""")

code("""# 可视化词表增长和压缩率
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 不同 merge 次数下的词表大小
merge_counts = [5, 10, 15, 20, 30, 50]
vocab_sizes = []
for n_merges in merge_counts:
    bpe_test = SimpleBPE(num_merges=n_merges)
    bpe_test.train(corpus)
    vocab_sizes.append(len(bpe_test.vocab))

axes[0].plot(merge_counts, vocab_sizes, 'b-o', linewidth=2.5, markersize=8)
axes[0].set_xlabel('合并次数')
axes[0].set_ylabel('词表大小')
axes[0].set_title('BPE: 词表随合并次数增长')
axes[0].grid(True, alpha=0.3)

# 压缩率: 原始字符数 / token 数
test_text = "low lower lowest newest newer wider widest"
char_count = len(test_text.replace(" ", ""))
compression = {}
for label, tokenizer in [('BPE', bpe), ('WordPiece', wp), ('Unigram', unigram)]:
    total_tokens = 0
    for word in test_text.split():
        total_tokens += len(tokenizer.encode(word))
    compression[label] = char_count / total_tokens

axes[1].bar(compression.keys(), compression.values(), color=['blue', 'green', 'orange'], alpha=0.7)
axes[1].set_ylabel('压缩率 (字符数/token数)')
axes[1].set_title('压缩率对比')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/fig_tokenizer_comparison.png', bbox_inches='tight')
plt.show()
print("压缩率越高, 序列越短, 模型处理越高效。")""")

# ============================================================
md("""## 6. 现代 LLM 的 Tokenizer

### 6.1 GPT 系列的 BPE

GPT-2/3/4 用 **tiktoken**（BPE 的优化实现）：
- 在**字节级**做 BPE（不是字符级）→ 天然支持所有 UTF-8 字符
- 词表大小：GPT-2 50K，GPT-4 100K

### 6.2 tiktoken 演示""")

code("""# 尝试使用 tiktoken (如果安装了)
try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-2")
    text = "Hello, world! Tokenization is unbelievably important."
    tokens = enc.encode(text)
    print(f"GPT-2 tokenizer:")
    print(f"  文本: {text}")
    print(f"  Token IDs: {tokens}")
    print(f"  Token 数: {len(tokens)}")
    print(f"  解码: {enc.decode(tokens)}")
    print(f"  词表大小: {enc.n_vocab}")
    has_tiktoken = True
except ImportError:
    print("tiktoken 未安装, 用模拟演示")
    has_tiktoken = False

# 模拟字节级 BPE 的概念
print("\\n字节级 BPE 的优势:")
text = "你好世界 🌍"
byte_seq = text.encode('utf-8')
print(f"  文本: {text}")
print(f"  UTF-8 字节: {list(byte_seq)}")
print(f"  字节数: {len(byte_seq)}")
print("  字节级 BPE: 任何文本都是字节序列 → 永远没有 OOV, 支持所有语言和 emoji!")""")

# ============================================================
md("""## 7. 小结与延伸

### ✅ 本章你掌握了

| 概念 | 状态 |
|------|------|
| 字符级 vs 词级的问题 | ✅ |
| 子词：两全其美的方案 | ✅ |
| BPE：反复合并最高频对 | ✅ |
| WordPiece：似然增益选择 | ✅ |
| Unigram：概率删除 + Viterbi 分词 | ✅ |
| 字节级 BPE：支持所有 UTF-8 | ✅ |

### 核心 takeaway

> **Tokenizer 是 LLM 的第一层——它决定了模型看到的世界**。
> BPE 是当前主流（GPT/Llama），WordPiece 用于 BERT，Unigram 用于 T5。
> 字节级 BPE 让模型天然支持多语言和 emoji——这是 GPT-4 的选择。

### 🔗 下一章预告

**`10_pretraining_scaling.ipynb`** — 自监督预训练(MLM/CLM/MAE) + Scaling law

---

> 💬 **写在最后**：tokenizer 看似简单的预处理，但词表设计、OOV 处理、多语言支持
> 都深刻影响模型能力。现代 LLM 的 tokenizer 是精心调优的工程产物。""")

# ============================================================
output_path = "notebooks/09_tokenization.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"✅ Notebook 已生成: {output_path}")
print(f"   共 {len(nb.cells)} 个 cell")