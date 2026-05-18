# Evaluation

This folder contains the FAQ-based benchmark evaluation pipeline for the eGov RAG assistant.

## Directory Structure

```text
evaluation/
├── crawlers/                     # FAQ data collection
│   ├── crawl_dvc_faq_ids.py      # Crawl FAQ pages by scanning ID range
│   └── parse_dvc_faq_detail.py   # Parse FAQ detail page HTML
├── utils/                        # Shared utilities
│   ├── text_normalize.py         # Text & title normalization (NFC, HTML unescape)
│   ├── title_matching.py         # Exact + fuzzy (RapidFuzz) title matching
│   └── jsonl_io.py               # JSONL/JSON read/write helpers
├── testsets/                     # Test data (JSONL files)
├── reports/                      # Generated benchmark reports
├── clean_dvc_faq_testset.py      # Clean raw FAQ, match titles to corpus
├── export_faq_eval_testset.py    # Export final 3-field testset
├── validate_faq_testset.py       # Validate testset integrity
├── eval_retrieval_title.py       # Retrieval evaluation (title matching)
├── eval_generation_judge.py      # Generation evaluation (LLM-as-judge)
├── eval_latency_dataset.py       # Latency benchmark on FAQ testset
└── run_faq_benchmark.py          # Orchestrator: runs all benchmarks
```

## Quick Start

### 1. Crawl FAQ data

```bash
python evaluation/crawlers/crawl_dvc_faq_ids.py \
    --start-id 15000 --end-id 30000 --max-valid 300 \
    --output evaluation/testsets/dvc_faq_raw.jsonl
```

> **Tip:** IDs below 15000 are mostly deleted. Start from 15000 for best hit rate.

### 2. Clean and match titles

```bash
python evaluation/clean_dvc_faq_testset.py \
    --input evaluation/testsets/dvc_faq_raw.jsonl \
    --corpus static/data/toan_bo_du_lieu_final.json \
    --output evaluation/testsets/dvc_faq_clean_full.jsonl \
    --manual-review-output evaluation/testsets/dvc_faq_manual_review.csv
```

### 3. Export final testset

```bash
python evaluation/export_faq_eval_testset.py \
    --input evaluation/testsets/dvc_faq_clean_full.jsonl \
    --output evaluation/testsets/dvc_faq_qa_500.jsonl \
    --limit 500 --seed 42
```

### 4. Validate

```bash
python evaluation/validate_faq_testset.py \
    --testset evaluation/testsets/dvc_faq_qa_500.jsonl
```

### 5. Run all benchmarks

```bash
# Requires API server running in another terminal: python scripts/run_dev.py
python evaluation/run_faq_benchmark.py \
    --testset evaluation/testsets/dvc_faq_qa_500.jsonl \
    --base-url http://localhost:7860 \
    --generation-limit 50
```

### 6. Run individual benchmarks

```bash
# Retrieval (no server needed — loads resources directly)
python evaluation/eval_retrieval_title.py --mode bm25
python evaluation/eval_retrieval_title.py --mode dense
python evaluation/eval_retrieval_title.py --mode hybrid

# Generation (requires running API server + GOOGLE_API_KEY)
python evaluation/eval_generation_judge.py --base-url http://localhost:7860 --limit 50

# Latency (requires running API server)
python evaluation/eval_latency_dataset.py --base-url http://localhost:7860 --limit 100
```

## Testset Schema

Each final test sample contains exactly 3 fields:

```json
{"question": "...", "reference_answer": "...", "expected_procedure_title": "..."}
```

## Metrics

### Retrieval
- **Recall@k** (k=1,3,5,10): Fraction of queries where the correct procedure appears in top-k results.
- **MRR@10**: Mean Reciprocal Rank — average of 1/rank of the first correct result.
- **nDCG@10**: Normalized Discounted Cumulative Gain.

### Generation (LLM-as-Judge)
- **Correctness** (1-5): Does the answer cover the key points from the reference?
- **Faithfulness** (1-5): Is the answer consistent with the reference (no contradictions)?
- **Pass rate**: Fraction of answers scoring ≥4 on both correctness and faithfulness.
- **Hallucination rate**: Fraction of answers containing information that contradicts the reference.
- **Source title match rate**: Fraction of responses where the correct procedure appears in source cards.

### Latency
- **p50, p90, p95, p99** (ms): End-to-end API response time percentiles.

## Notes

- Use `GENAI_MODEL=gemini-1.5-flash` in `.env` for benchmarking (1,500 req/day free tier). The default `gemini-2.5-flash` has only 20 req/day on free tier.
- FAQ reference answers are NOT indexed into the retrieval corpus — they are used solely for generation quality evaluation.
- Retrieval metrics use exact normalized title matching (NFC + lowercase + punctuation removal).
