# Jev-like Transformer

### A Parallel Decision Model Based on Shared State and Dynamic Candidate Sets

A **Jev-like Transformer architecture** for parallel decision making.

Instead of generating answers autoregressively, the model processes multiple independent Questions under a shared State and directly scores their candidate answers.

```text
Traditional LLM

State
  ↓
Token Generation
  ↓
token → token → token → ... → answer
```

```text
Jev-like Transformer

                 Shared State
                      │
                      ▼
                 Transformer
              ┌───────┼───────┐
              ▼       ▼       ▼
             Q1      Q2      Q3
              │       │       │
              ▼       ▼       ▼
             h1      h2      h3
              │       │       │
              ▼       ▼       ▼
         Candidates Candidates Candidates
              │       │       │
              ▼       ▼       ▼
            Scorer  Scorer  Scorer
              │       │       │
              ▼       ▼       ▼
             P1      P2      P3
```

## Key Idea

The core architecture consists of four components:

```text
Shared State
+
Parallel Questions
+
Dynamic Candidates
+
Shared Scorer
```

The key idea is:

> **The Head no longer determines the output dimension; the Candidate Set determines the output dimension.**

Instead of mapping a hidden state to a fixed vocabulary-sized output, the model embeds the candidate answers and scores only the candidates relevant to each Question.

```text
Question hidden h
        +
Candidate embeddings E
        ↓
   Shared Scorer
        ↓
     logits[K]
        ↓
     softmax
        ↓
   probabilities
```

---

## Architecture

Let the shared context be:

```text
S = (s₁, s₂, …, sₘ)
```

and the i-th Question be:

```text
Qᵢ = (qᵢ₁, qᵢ₂, …, qᵢₖᵢ)
```

The Transformer produces a representation for each Question:

```text
Hₛ = Transformer(S)

hᵢ ∈ Rᵈ
```

The candidate scoring function is:

```text
zᵢⱼ = f(hᵢ, eᵢⱼ)
```

A simple implementation uses scaled dot-product scoring:

```text
zᵢⱼ = hᵢ · eᵢⱼ / √d
```

followed by:

```text
P(aᵢⱼ | S, Qᵢ) = softmaxⱼ(zᵢⱼ)
```

Each Question can therefore have its own candidate set:

```text
Q1 → 2 candidates
Q2 → 5 candidates
Q3 → 100 candidates
```

The Transformer architecture itself does not need to change.

---

## Attention Structure

The first implementation uses an Attention Mask that allows each Question to attend to the shared State and itself, but not to other Questions.

```text
             State   Q1   Q2   Q3

State          ✓     ✓    ✓    ✓
Q1             ✓     ✓    ✗    ✗
Q2             ✓     ✗    ✓    ✗
Q3             ✓     ✗    ✗    ✓
```

This allows multiple Questions to be processed in a single Transformer forward pass.

Conceptually:

```text
                     Shared State
                          │
                          ▼
                    Transformer
                  ┌───────┼───────┐
                  │       │       │
                 Q1      Q2      Q3
                  │       │       │
                  ▼       ▼       ▼
                 h1      h2      h3
```

---

## Dynamic Candidate Scoring

Traditional LLMs typically operate over a fixed vocabulary:

```text
V ≈ 10⁴ ~ 10⁵
```

The Jev-like approach instead defines a Question-specific candidate space:

```text
Aᵢ = {aᵢ₁, aᵢ₂, ..., aᵢₖᵢ}
```

The number and content of candidates can differ between Questions.

For example:

```text
Q1:
Should a refund be issued?

Candidates:
{Yes, No}
```

```text
Q2:
What type of user?

Candidates:
{Normal User,
 High-Value User,
 Risk User,
 Blacklisted User}
```

```text
Q3:
Which tool should be called?

Candidates:
{search_web,
 database,
 calculator,
 send_email,
 browser,
 ...}
```

This makes the output space dynamic rather than fixed.

---

## Candidate Sources

### 1. Fixed Candidate Vocabulary

Candidates can come from a predefined embedding table.

```text
["yes", "no"]
```

### 2. Natural Language Candidates

Candidates can be encoded from their text descriptions.

```text
Question:
Which action should the customer take?

Candidates:
refund
exchange
reject
manual review
```

Each candidate is converted into an embedding and scored against the Question representation.

### 3. Runtime Dynamic Candidates

For Agent tool selection, the candidate set can be generated dynamically at runtime.

```text
Question:
Which tool should be called next?

Candidates:
search_web
database
calculator
send_email
browser
```

The model only scores the currently available candidates.

---

## Minimal PyTorch Scorer

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

Then:

```python
logits = scorer(h, candidates)

probs = torch.softmax(logits, dim=-1)
```

Different Questions can have different numbers of candidates.

For batching, candidate sets can be padded and masked:

```text
Q1 → K=2
Q2 → K=5
Q3 → K=17
```

```python
logits = logits.masked_fill(~candidate_mask, -inf)
```

---

## More Powerful Scorers

The simplest scorer is a dot product, but more expressive functions can be used.

### MLP

```text
z = f([h; e])
```

```text
h/e concat
    ↓
   MLP
    ↓
 scalar
```

### Bilinear

```text
z = hᵀ W e
```

```python
logit = h @ W @ e
```

Another option is to independently transform the Question and Candidate embeddings with MLPs and then compute their dot product.

---

## Training

A training sample consists of:

```text
State
Question
Candidates
Correct Answer
```

For a single Question:

```text
L = -log P(a* | S, Q)
```

For multiple independent Questions under the same State:

```text
State S

Q1 → A1
Q2 → A2
...
QN → AN
```

The total loss is:

```text
L = Σᵢ Lᵢ
```

This allows multiple independent decision problems to be solved under the same State.

---

## Applications

The architecture is designed for tasks such as:

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

A particularly relevant scenario is:

```text
One shared State
        +
Many independent Questions
        +
Different candidate sets
```

---

## Difference from Traditional LLMs

|                    | Traditional LLM                 | Jev-like Model                        |
| ------------------ | ------------------------------- | ------------------------------------- |
| Output unit        | Token                           | Candidate                             |
| Output space       | Fixed Vocabulary                | Question-specific                     |
| Inference          | Autoregressive                  | Parallel decision                     |
| Multiple questions | Repeated generation             | Batch parallel                        |
| State              | Participates in each generation | Shared KV                             |
| Output length      | Variable token sequence         | Number of candidates                  |
| Main task          | Text generation                 | Decision / Classification / Selection |

The fundamental change is:

```text
Token Generation
        ->
Candidate Scoring
```

This is not simply adding multiple classification heads to an LLM.

---

## Agent Decision Model

A natural extension is:

```text
                     State
                       │
                       ▼
                  Transformer
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
     Tool Q         Action Q        Risk Q
        │              │              │
        ▼              ▼              ▼
   Candidates      Candidates      Candidates
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                     Score
```

For example:

```text
search_web   0.71
database     0.18
calculator   0.07
browser      0.04
```

This suggests a possible architecture:

```text
LLM
├── Reasoning / Generation
└── Decision Transformer
    ├── Classification
    ├── Routing
    ├── Tool Selection
    ├── Ranking
    └── Agent Control
```

---

## Roadmap

### V0 — Minimal Prototype

```text
Qwen/Llama
   ↓
State + Question
   ↓
Transformer
   ↓
Question hidden
   ↓
Candidate embedding
   ↓
Dot product
   ↓
Softmax
```

Goal:

> First prove that the model can perform the task.

### V1 — Parallel Questions

Multiple Questions → one batch forward → validate throughput.

### V2 — Shared State KV

Multiple Questions share the State KV Cache.

### V3 — Runtime Dynamic Candidates

Support dynamically generated Candidate Sets.

### V4 — Inference Optimization

Potential optimizations:

* Paged KV Cache
* FlashAttention
* Continuous Batching
* Candidate Embedding Cache
* Tensor Parallel
* GPU fused scorer

The long-term goal is a high-speed Decision Model designed for the Agent Decision Loop.

---

## Project Status

This repository currently presents the architecture and the initial implementation direction.

The first implementation will be built on an existing Qwen/Llama Transformer rather than redesigning the Transformer backbone.

The initial system only requires:

```text
Attention Mask
+
Question Batch
+
Candidate Embedding
+
Shared Scorer
```

Experimental results and implementation details will be added as development progresses.

---

## Paper

The full technical description is available in:

```text
paper/jev-like-transformer.pdf
```

**Jev-like Transformer: A Parallel Decision Model Based on Shared State and Dynamic Candidate Sets**

---

## Core Architecture

```text
S + {Q₁, ..., Qₙ}
        ->
{h₁, ..., hₙ}
        ->
Candidate Scoring
        ->
{P₁, ..., Pₙ}
```

For each Question:

```text
Pᵢ = softmax(f(hᵢ, Eᵢ))
```

where:

```text
|E₁|, |E₂|, ..., |Eₙ|
```

can all be different.

---

## Citation

If you find this architecture useful, please cite the paper:

```bibtex
@article{jev_like_transformer,
  title   = {Jev-like Transformer: A Parallel Decision Model Based on Shared State and Dynamic Candidate Sets},
  author  = {Your Name},
  year    = {2026},
  note    = {Technical proposal}
}
```

---

## License

See [LICENSE](LICENSE).
