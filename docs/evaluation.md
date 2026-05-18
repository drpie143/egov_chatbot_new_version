# Evaluation

## Overview

eGov-Bot is evaluated using an FAQ-based benchmark derived from the Vietnamese National Public Service Portal (dichvucong.gov.vn). This approach uses real user-style questions with official reference answers, providing a rigorous assessment of both retrieval quality and answer generation.

## Benchmark Design

The benchmark consists of three layers:

### Layer A — Retrieval Ablation (Title Matching)

Measures whether the retriever returns the expected administrative procedure in the top-k results. Each retrieval mode (BM25, Dense, Hybrid) is evaluated independently.

**Metrics:** Recall@1, Recall@3, Recall@5, Recall@10, MRR@10, nDCG@10

### Layer B — End-to-End Generation Quality

Uses an LLM-as-judge (Gemini) to score generated answers against official FAQ reference answers.

**Metrics:** Correctness (1-5), Faithfulness (1-5), Pass rate, Hallucination rate, Source title match rate

### Layer C — Latency Profiling

Measures end-to-end `/chat` API response time across the full testset with warm-up requests.

**Metrics:** p50, p90, p95, p99 latency (ms)

## Results Summary

### Retrieval Ablation

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|--------|----------|----------|-----------|--------|---------|
| BM25   | 0.7432   | 0.9054   | 0.9324    | 0.8220 | 0.8496  |
| Dense  | 0.8514   | 0.9324   | 0.9324    | 0.8829 | 0.8953  |
| Hybrid | 0.8514   | 0.9324   | 0.9459    | 0.8820 | 0.8975  |

### End-to-End Latency

| Percentile | Value |
|------------|------:|
| p50 | 4,819 ms |
| p95 | 11,881 ms |

> Full report: `evaluation/reports/faq_latest_report.md`

## Running Benchmarks

```bash
# Full pipeline (requires running API server in another terminal)
python evaluation/run_faq_benchmark.py \
    --testset evaluation/testsets/dvc_faq_qa_500.jsonl \
    --base-url http://localhost:7860 \
    --generation-limit 50

# Individual retrieval ablation (no server needed)
python evaluation/eval_retrieval_title.py --mode bm25
python evaluation/eval_retrieval_title.py --mode dense
python evaluation/eval_retrieval_title.py --mode hybrid

# Generation only
python evaluation/eval_generation_judge.py --base-url http://localhost:7860 --limit 50

# Latency only
python evaluation/eval_latency_dataset.py --base-url http://localhost:7860 --limit 100
```

Reports are generated in `evaluation/reports/`.

## Important Notes

- FAQ reference answers are used only for generation quality evaluation — they are NOT indexed into the retrieval corpus.
- Retrieval metrics use exact normalized title matching (NFC + lowercase + punctuation removal) for reproducibility.
- LLM-as-judge requires a `GOOGLE_API_KEY` with sufficient quota. Use `GENAI_MODEL=gemini-1.5-flash` for higher free-tier limits (1,500 req/day vs 20 req/day for gemini-2.5-flash).
- See `evaluation/README.md` for the full data collection pipeline.
