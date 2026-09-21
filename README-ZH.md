# Jev-like Transformer

## 基于共享状态与动态候选集的并行决策模型

Jev-like Transformer 是一种面向**并行决策任务**的 Transformer 架构。

它的核心思想是：

> **Shared State + Parallel Questions + Dynamic Candidates + Shared Scorer**

传统 LLM 主要采用自回归生成：

```text
Context
   ↓
Transformer
   ↓
Token 1 → Token 2 → Token 3 → ... → Answer
```

当同一个 State 下存在大量相互独立的问题时，每个问题都进行独立的生成，会产生大量重复计算。

Jev-like Transformer 将问题改造成并行决策：

```text
                    Shared State
                         │
                         ▼
                    Transformer
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
             Q1         Q2         Q3
              │          │          │
              ▼          ▼          ▼
             h1         h2         h3
              │          │          │
              ▼          ▼          ▼
        Candidates  Candidates  Candidates
              │          │          │
              ▼          ▼          ▼
           Scorer      Scorer      Scorer
              │          │          │
              ▼          ▼          ▼
             P1         P2         P3
```

---

## 核心特点

### 1. Shared State

所有 Question 共享同一个 State。

公共上下文只需要进行一次主要计算，并可以通过共享 KV Cache 供多个 Question 使用。

### 2. Parallel Questions

多个 Question 在一次 Transformer forward 中并行计算。

每个 Question 最终得到自己的语义表示：

```text
Q1 → h1
Q2 → h2
Q3 → h3
...
Qn → hn
```

### 3. Dynamic Candidate Set

每个 Question 可以拥有不同数量、不同内容的候选答案。

例如：

```text
Q1 → {Yes, No}
       ↓
      2 logits

Q2 → {Normal, High-Value, Risk, Blacklisted}
       ↓
      4 logits

Q3 → {Candidate 1, ..., Candidate 100}
       ↓
      100 logits
```

因此，输出空间不再由固定的 Prediction Head 决定，而是由 Candidate Set 动态决定。

### 4. Shared Candidate Scorer

不为每一种候选数量设计不同的 Head，而是将 Candidate 转换成 embedding，然后使用统一的 scorer 进行匹配。

最简单的实现：

```text
zᵢⱼ = hᵢ · eᵢⱼ / √d
```

然后：

```text
P(aᵢⱼ | S, Qᵢ) = softmaxⱼ(zᵢⱼ)
```

因此：

```text
Candidate Set
      ↓
Candidate Embedding
      ↓
Shared Scorer
      ↓
Variable-length Logits
      ↓
Softmax
      ↓
Probability Distribution
```

---

## 与传统 LLM 的区别

|          | 传统 LLM          | Jev-like Transformer                  |
| -------- | --------------- | ------------------------------------- |
| 输出单位     | Token           | Candidate                             |
| 输出空间     | 固定 Vocabulary   | Question-specific                     |
| 推理方式     | Autoregressive  | Parallel Decision                     |
| 多问题处理    | 重复生成            | Batch Parallel                        |
| State    | 每次生成参与计算        | Shared                                |
| KV Cache | 每个生成过程使用        | 可共享 State KV                          |
| 输出长度     | Token Sequence  | Candidate 数量                          |
| 主要任务     | Text Generation | Decision / Classification / Selection |

核心变化不是简单地：

```text
给 LLM 增加几个 Classification Head
```

而是改变 Transformer 的输出范式：

```text
Token Generation
        ↓
Candidate Scoring
```

---

## 模型结构

定义：

```text
S = (s₁, s₂, …, sₘ)

Qᵢ = (qᵢ₁, qᵢ₂, …, qᵢₖᵢ)
```

共享 State：

```text
Hₛ = Transformer(S)
```

每个 Question 得到：

```text
hᵢ ∈ Rᵈ
```

候选答案：

```text
eᵢⱼ ∈ Rᵈ
```

Candidate Scoring：

```text
zᵢⱼ = f(hᵢ, eᵢⱼ)
```

最简单的 scorer：

```text
zᵢⱼ = hᵢ · eᵢⱼ / √d
```

最终：

```text
P(aᵢⱼ | S, Qᵢ) = softmaxⱼ(zᵢⱼ)
```

---

## Attention Mask

第一版不需要修改 Transformer Backbone。

可以直接使用现有的 HuggingFace Transformer，并通过 Attention Mask 实现多个 Question 的并行计算。

例如：

```text
             State   Q1   Q2   Q3

State          ✓     ✓    ✓    ✓

Q1             ✓     ✓    ✗    ✗

Q2             ✓     ✗    ✓    ✗

Q3             ✓     ✗    ✗    ✓
```

含义：

* Question 可以看到 State
* Question 可以看到自己的 Token
* Question 不需要看到其他 Question

因此多个 Question 可以在一次 Transformer forward 中同时得到表示。

---

## Candidate Scorer

最简单的 PyTorch 实现：

```python
class CandidateScorer(nn.Module):
    def forward(self, h, candidate_embeddings):
        # h: [B, D]
        # candidate_embeddings: [B, K, D]

        logits = torch.einsum(
            "bd,bkd->bk",
            h,
            candidate_embeddings
        )

        return logits
```

然后：

```python
logits = scorer(h, candidates)

probs = torch.softmax(logits, dim=-1)
```

不同 Question 可以拥有不同数量的 Candidate。

例如：

```text
Q1: K = 2
Q2: K = 5
Q3: K = 17
```

Batch 中可以通过 padding + mask 处理：

```python
logits = logits.masked_fill(
    ~candidate_mask,
    -inf
)
```

---

## 更强的 Scorer

除了简单的 dot product，还可以使用更加复杂的 scorer：

### MLP

```text
z = f([h; e])
```

```text
h + e
  ↓
Concat
  ↓
MLP
  ↓
Scalar Logit
```

### Bilinear

```text
z = hᵀ W e
```

PyTorch：

```python
logit = h @ W @ e
```

也可以分别对 Question embedding 和 Candidate embedding 使用 MLP，再进行 dot product。

---

## Candidate 的来源

### A. 固定 Candidate Vocabulary

例如：

```text
["yes", "no"]
```

直接建立 Candidate embedding table。

---

### B. Natural Language Candidate

Candidate 本身可以是自然语言。

例如：

```text
Question:
客户应该采取什么操作？

Candidates:
- 退款
- 换货
- 拒绝
- 人工审核
```

将 Candidate 文本编码成 embedding，然后进行 scoring。

---

### C. Runtime Dynamic Candidate

Candidate Set 可以在运行时动态产生。

例如 Agent Tool Selection：

```text
Question:
下一步调用哪个工具？

Candidates:
- search_web
- database
- calculator
- send_email
- browser
```

模型只需要对当前 Candidate Set 进行评分。

---

## 应用场景

该架构尤其适合：

* Classification
* Routing
* Agent Decision
* Tool Selection
* Risk Assessment
* Intent Recognition
* Multi-condition Decision
* Ranking
* Structured Decision
* Agent Action Selection

特别适合下面这种场景：

```text
一个 State
    +
大量独立 Questions
    +
每个 Question 拥有不同数量、不同内容的 Candidates
```

---

## Training

一个训练样本可以表示为：

```text
State
Question
Candidates
Correct Answer
```

例如：

```text
State:
Customer requests a refund.

Question:
Is the refund allowed?

Candidates:
Yes / No

Label:
Yes
```

Loss：

```text
L = -log P(a* | S, Q)
```

同一个 State 下可以同时存在多个 Question：

```text
S

Q1 → A1
Q2 → A2
Q3 → A3
...
QN → AN
```

总 Loss：

```text
L = Σᵢ Lᵢ
```

---

## 最小可行实现

### V0

先证明基本任务能够完成：

```text
Qwen / Llama
      ↓
State + Question
      ↓
Transformer
      ↓
Question Hidden
      ↓
Candidate Embedding
      ↓
Dot Product
      ↓
Softmax
```

### V1

多个 Question 一次 Batch Forward：

```text
Q1
Q2
Q3
...
Qn
 ↓
One Transformer Forward
```

验证并行计算的吞吐提升。

### V2

引入 State KV Cache：

```text
State
  ↓
Shared KV Cache
  ├── Q1
  ├── Q2
  ├── Q3
  └── Qn
```

多个 Question 共享 State KV。

### V3

支持 Runtime Dynamic Candidate。

### V4

进一步进行推理优化：

* Paged KV Cache
* FlashAttention
* Continuous Batching
* Candidate Embedding Cache
* Tensor Parallel
* GPU Fused Scorer

最终形成面向 **Agent Decision Loop** 的高速 Decision Model。

---

## Architecture Overview

```text
                         STATE
                           │
                           ▼
                   Transformer Encode
                           │
                           ▼
                   Shared KV Cache
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
        Question 1     Question 2     Question 3
            │              │              │
            ▼              ▼              ▼
       Transformer     Transformer     Transformer
          Query            Query           Query
            │              │              │
            ▼              ▼              ▼
           h1             h2             h3
            │              │              │
            ▼              ▼              ▼
       Candidates      Candidates      Candidates
            │              │              │
            ▼              ▼              ▼
       Embeddings      Embeddings      Embeddings
            │              │              │
            ▼              ▼              ▼
          Scorer          Scorer          Scorer
            │              │              │
            ▼              ▼              ▼
         logits[k1]     logits[k2]     logits[k3]
            │              │              │
            ▼              ▼              ▼
         softmax         softmax         softmax
            │              │              │
            ▼              ▼              ▼
           P1             P2             P3
```

---

## Core Contributions

### 1. Shared State

同一个公共上下文只进行一次主要计算。

### 2. Parallel Questions

多个 Question 在同一个 Transformer inference 中并行处理。

### 3. Dynamic Candidate Set

每个 Question 拥有自己的 Candidate Set，输出维度由 Candidate Set 动态决定。

### 4. Shared Candidate Scorer

不需要针对不同 Candidate 数量设计不同 Prediction Head。

使用：

```text
Candidate Embedding
        +
Shared Scorer
```

统一完成候选答案评分。

整体流程：

```text
S + {Q₁, ..., Qₙ}
        ↓
{h₁, ..., hₙ}
        ↓
Candidate Scoring
        ↓
{P₁, ..., Pₙ}
```

其中：

```text
Pᵢ = softmax(f(hᵢ, Eᵢ))
```

并且：

```text
|E₁|, |E₂|, ..., |Eₙ|
```

可以完全不同。

---

## Future Direction

进一步可以形成：

```text
                         LLM
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
     Reasoning / Generation     Decision Transformer
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
              Classification       Routing        Tool Selection
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       ▼
                                   Ranking
                                       │
                                       ▼
                                Agent Control
```

对于 Agent 系统，可以直接输出：

```text
search_web   0.71
database     0.18
calculator  0.07
browser      0.04
```

而不是先生成一段文本，再从文本中解析下一步动作。

---

## Core Idea

最终可以将整个架构概括为：

```text
Shared State
      +
Parallel Questions
      +
Dynamic Candidates
      +
Shared Scorer
```

核心原则：

> **Head 不再决定输出维度，Candidate Set 决定输出维度。**

这使得一个统一的 Transformer 可以在同一个 State 下，同时处理具有不同数量、不同内容候选答案的多个决策问题。

---

## License

This project is licensed under the **Apache License 2.0**.

See [LICENSE](LICENSE) for details.
