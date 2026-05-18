# eGov-Bot FAQ Benchmark Report

Generated: 2026-05-17T00:16:37

## 1. Testset

- **Source:** Official FAQ-style questions from the Vietnamese National Public Service Portal (dichvucong.gov.vn).
- **Schema:** `question`, `reference_answer`, `expected_procedure_title`
- **Size:** 74 samples (after crawling, cleaning, title matching, and deduplication)
- **Testset path:** `evaluation/testsets/dvc_faq_qa_500.jsonl`

> Note: FAQ reference answers are used for generation quality evaluation only. They are NOT indexed into the retrieval corpus.

## 2. Retrieval Results (Ablation Study)

Evaluated on 74 FAQ questions. The retriever must return the correct procedure title in top-k results.

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|--------|----------|----------|----------|-----------|--------|---------|
| BM25   | 0.7432   | 0.8919   | 0.9054   | 0.9324    | 0.8220 | 0.8496  |
| Dense  | 0.8514   | 0.9054   | 0.9324   | 0.9324    | 0.8829 | 0.8953  |
| Hybrid | 0.8514   | 0.9054   | 0.9324   | 0.9459    | 0.8820 | 0.8975  |

**Key findings:**
- Dense retrieval significantly outperforms BM25 at Recall@1 (+10.8pp), demonstrating the semantic embedding model's effectiveness for Vietnamese administrative text.
- Hybrid retrieval achieves the highest Recall@10 (94.59%) by combining BM25's keyword coverage with Dense's semantic understanding.
- MRR@10 is marginally lower for Hybrid vs Dense (0.8820 vs 0.8829) — a known RRF (Reciprocal Rank Fusion) trade-off where BM25 keyword matches can sometimes perturb top-1 ranking.
- All three methods exceed 90% Recall@5, indicating the procedure corpus and retrieval pipeline are well-suited for this FAQ domain.

## 3. Generation Quality (LLM-as-Judge)

Evaluated on 74 FAQ questions using the `/chat` API endpoint. An LLM judge (Gemini) scores each model answer against the official FAQ reference answer on correctness (1-5), faithfulness (1-5), and hallucination detection.

| Metric | Value |
|--------|------:|
| API success rate | 100% |
| Source title match rate | 94.59% |
| Latency p50 | 4,819 ms |
| Latency p95 | 11,881 ms |

> **Note on LLM-as-Judge scores:** The Gemini judge scoring was rate-limited during this benchmark run (free-tier quota: 20 requests/day for gemini-2.5-flash). Only 13 of 20 sampled questions received judge scores. Per-sample judge results are available in `evaluation/reports/faq_generation_per_sample.jsonl`. Re-run with `GENAI_MODEL=gemini-1.5-flash` (1,500 req/day limit) for full coverage.

## 4. End-to-End Latency

Measured over 74 FAQ questions with 2 warm-up requests. Includes retrieval + generation time.

| Percentile | Latency |
|------------|--------:|
| p50        | 4,819 ms |
| p90        | 9,940 ms |
| p95        | 11,881 ms |
| p99        | 13,017 ms |
| Mean       | 5,892 ms |

> Latency is dominated by the Gemini API call (~3-10s). Retrieval-only latency is under 200ms for Dense and ~3s for BM25/Hybrid.

## 5. Pipeline Steps

| Step | Status |
|------|--------|
| Testset validation | ✓ |
| Retrieval — BM25 | ✓ |
| Retrieval — Dense | ✓ |
| Retrieval — Hybrid | ✓ |
| Generation — LLM-as-Judge | ✓ (partial, rate-limited) |
| Latency profiling | ✓ |

## 6. Reproduce

```bash
# Start API server
python scripts/run_dev.py

# Run full benchmark (in another terminal)
python evaluation/run_faq_benchmark.py \
    --testset evaluation/testsets/dvc_faq_qa_500.jsonl \
    --base-url http://localhost:7860 \
    --generation-limit 50
```
