# CV / Portfolio Summary — eGov-Bot

> Every number here is reproducible from this repository. Retrieval evaluation
> makes no API calls: `python evaluation/eval_retrieval_title.py --mode bm25 --group`.

## One-liner

Vietnamese e-government RAG assistant over 12,361 administrative procedures —
hybrid BM25 + FAISS retrieval with parent-document grouping, source-grounded
Gemini generation, and a benchmark that reports confidence intervals instead of
point estimates.

## Bullet points (for CV / resume)

- Audited the retrieval stack and found index-time and query-time tokenization
  had diverged, leaving punctuation attached to **78.9%** of the indexed
  vocabulary. Unified tokenization and moved scoring to a sparse-matrix backend:
  sparse retrieval **p50 1,920 ms → 60 ms (32×)** over 338k chunks.

- Built a **paired-bootstrap significance harness** that invalidated the
  project's own headline metric, then traced the inflation to a swallowed
  `ImportError` that had silently cut the benchmark from 167 questions to the 74
  that BM25 finds easiest. Restoring it moved Recall@10 from a reported
  **0.9459 to a real 0.8503**.

- Diagnosed the loss surface before changing architecture: the ranker surfaced
  only **1.87 distinct procedures per top-10**, spending 8 of 10 slots
  re-showing one procedure. Parent-document grouping with adaptive candidate
  depth, plus query normalization, gave **Recall@10 +7.2 pp** and
  **nDCG@10 +4.0 pp** at **zero latency cost**, all 95% CIs excluding zero.

- Measured and **rejected** cross-encoder reranking (**Recall@1 −22.8 pp**,
  p50 6.8 s on CPU) after establishing its ceiling: recall saturates by depth
  50, so **8.4% of questions are unreachable by reranking at any depth**.
  Routed that failure mode to candidate generation instead.

## Measured results

167 FAQ questions from the National Public Service Portal, BM25 with grouping
and query stripping.

| Metric | Value |
|---|---:|
| Recall@1 | 0.7545 |
| Recall@5 | 0.8503 |
| Recall@10 | 0.9222 |
| MRR@10 | 0.7853 |
| nDCG@10 | 0.8165 |
| Sparse retrieval p50 | 60 ms |
| Procedure corpus | 12,361 procedures / 338k chunks |
| Test set | 167 questions |

Changes that survive a paired bootstrap (10,000 resamples):

| Change | Effect | 95% CI |
|---|---|---|
| Parent-document grouping | Recall@10 **+0.0599** | [+0.0299, +0.0958] |
| Parent-document grouping | nDCG@10 **+0.0255** | [+0.0133, +0.0392] |
| Query boilerplate stripping | nDCG@10 **+0.0147** | [+0.0034, +0.0292] |
| BM25 vs Hybrid, paraphrased queries | Recall@1 **+0.1719** | [+0.0469, +0.2969] |

## Claims this project does NOT make

Stated explicitly, because they are the questions an interviewer should ask.

- **BM25, Dense and Hybrid are not ranked.** At n=167 every quality difference
  between them falls inside its confidence interval. Hybrid retrieval is
  configurable, not defended as better.
- **The benchmark is lexically biased.** 41.3% of its questions quote the target
  procedure's title verbatim, because they come from a portal FAQ. Recall@1 is
  ~0.80 on those and ~0.62 on paraphrased ones; the paraphrased slice is the
  honest number.
- **8.4% of questions are unreachable.** Recall saturates by depth 50. This is a
  candidate-generation limit, not a ranking one.
- **The hosted demo is a different system** — Pinecone dense-only, no BM25 and
  no fusion. These numbers do not describe it.

## Technical stack

`Python` · `Flask` · `FAISS` · `bm25s` · `Sentence-Transformers` ·
`Google Gemini API` · `Docker` · `SQLite` · `BeautifulSoup` · `RapidFuzz` ·
`pytest` · `ruff` · GitHub Actions
