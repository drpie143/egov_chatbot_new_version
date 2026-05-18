# CV / Portfolio Summary — eGov-Bot

## One-liner

Built a production-ready Vietnamese e-government RAG chatbot with hybrid BM25 + FAISS retrieval, source-grounded Gemini generation, and an automated FAQ-based evaluation pipeline achieving **Recall@10 = 94.59%** and **MRR@10 = 0.8975** on 74 real-world questions.

## Bullet Points (for CV / Resume)

- Designed and implemented a **Hybrid RAG pipeline** combining BM25 sparse search, FAISS dense retrieval (`AITeamVN/Vietnamese_Embedding`), and Reciprocal Rank Fusion, achieving **Recall@10 = 94.59%** and **nDCG@10 = 0.8975** on a 74-sample FAQ benchmark from the National Public Service Portal.
- Built an **automated 3-layer evaluation pipeline**: (1) Retrieval ablation with title matching (Recall@k, MRR, nDCG), (2) LLM-as-judge generation scoring (correctness, faithfulness, hallucination), (3) End-to-end latency profiling (p50 = 4.8s, p95 = 11.9s).
- Developed a **data collection pipeline** with automated web crawlers, NFC text normalization, exact + fuzzy (RapidFuzz) title matching against a 12,000+ procedure corpus, and JSONL testset validation.
- Deployed via **Docker Compose** with Flask API, SQLite feedback logging, multi-turn conversation context, and source-grounded answer generation using Google Gemini.

## Key Metrics

| Metric | Value |
|--------|------:|
| Hybrid Recall@1 | 85.14% |
| Hybrid Recall@5 | 93.24% |
| Hybrid Recall@10 | 94.59% |
| Hybrid MRR@10 | 0.8820 |
| Hybrid nDCG@10 | 0.8975 |
| Dense vs BM25 Recall@1 gain | +10.8pp |
| API success rate | 100% |
| Source title match rate | 94.59% |
| End-to-end latency p50 | 4,819 ms |
| Testset size | 74 FAQ samples |
| Procedure corpus | 12,361 procedures |

## Technical Stack

`Python` · `Flask` · `FAISS` · `BM25 (rank-bm25)` · `Sentence-Transformers` · `Google Gemini API` · `Docker` · `SQLite` · `BeautifulSoup` · `RapidFuzz`
