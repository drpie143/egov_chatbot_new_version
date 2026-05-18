"""End-to-end generation evaluation with LLM-as-judge.

Calls the /chat API, checks source title match, and uses an LLM judge to score
answer correctness, faithfulness, and hallucination against the reference answer.

Usage:
    python evaluation/eval_generation_judge.py \
        --base-url http://localhost:7860 \
        --testset evaluation/testsets/dvc_faq_qa_500.jsonl \
        --output evaluation/reports/faq_generation_metrics.json \
        --per-sample-output evaluation/reports/faq_generation_per_sample.jsonl \
        --limit 100 \
        --judge-provider gemini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.utils.jsonl_io import read_jsonl, write_json, write_jsonl  # noqa: E402
from evaluation.utils.text_normalize import normalize_title  # noqa: E402

JUDGE_PROMPT = """Bạn là evaluator cho hệ thống hỏi đáp thủ tục hành chính.

Nhiệm vụ: chấm câu trả lời của model dựa trên câu hỏi và câu trả lời tham chiếu. Lưu ý: Câu trả lời tham chiếu thường rất ngắn gọn, trong khi model RAG có thể trả lời chi tiết hơn dựa trên toàn bộ quy định của thủ tục.

Question:
{question}

Reference answer:
{reference_answer}

Model answer:
{model_answer}

Hãy đánh giá:
1. correctness_score: 1-5. Model có trả lời ĐÚNG trọng tâm câu hỏi và BAO HÀM được các ý chính của reference answer không?
2. faithfulness_score: 1-5. Thông tin model đưa ra có mâu thuẫn hoặc sai lệch với reference answer không? (Không phạt nếu model cung cấp thêm thông tin chi tiết hợp lệ).
3. missing_information: danh sách ý quan trọng trong reference mà model bị thiếu.
4. hallucinated_information: danh sách thông tin model đưa ra BỊ SAI LỆCH hoặc MÂU THUẪN hoàn toàn với reference. (Không tính các chi tiết bổ sung thêm là hallucinated).
5. final_verdict: "pass" hoặc "fail".

Quy tắc:
- THƯỞNG ĐIỂM (PASS): Nếu model cover được các ý của reference và trả lời đúng trọng tâm.
- KHÔNG PHẠT: Nếu model đưa ra thêm các bước, thêm hồ sơ chi tiết mở rộng so với reference (vì model đọc từ toàn văn thủ tục).
- PHẠT NẶNG (FAIL): Nếu model trả lời sai cơ quan có thẩm quyền, hoặc thông tin hoàn toàn trái ngược với reference.
- PASS khi correctness_score >= 4, faithfulness_score >= 4.

Chỉ trả về JSON hợp lệ, không markdown, không giải thích thêm."""


def _call_gemini_judge(question: str, reference_answer: str, model_answer: str) -> dict | None:
    """Call Gemini API as LLM judge."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai is not installed.", file=sys.stderr)
        return None

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY_2")
    if not api_key:
        print("No GOOGLE_API_KEY found for judge.", file=sys.stderr)
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(os.getenv("GENAI_MODEL", "gemini-1.5-flash"))

    prompt = JUDGE_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        model_answer=model_answer,
    )

    for attempt in range(4):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Try to extract JSON from response
            return _parse_judge_json(text)
        except Exception as exc:
            if attempt < 3:
                # Exponential backoff for rate limits: 5s, 15s, 45s
                sleep_time = 5 * (3 ** attempt)
                print(f"  [Judge Rate Limit/Error] Waiting {sleep_time}s before retry {attempt+1}... ({exc})", file=sys.stderr)
                time.sleep(sleep_time)
            else:
                print(f"  Judge API failed after {attempt+1} attempts: {exc}", file=sys.stderr)
                return None

    return None


def _parse_judge_json(text: str) -> dict | None:
    """Parse judge output, handling various formats."""
    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _source_title_match(sources: list[dict], expected_title: str) -> bool:
    """Check if expected procedure title appears in the source list."""
    expected_norm = normalize_title(expected_title)
    for source in sources:
        title = source.get("title", "")
        if normalize_title(title) == expected_norm:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation evaluation with LLM-as-judge.")
    parser.add_argument("--base-url", default="http://localhost:7860")
    parser.add_argument("--testset", default="evaluation/testsets/dvc_faq_qa_500.jsonl")
    parser.add_argument("--output", default="evaluation/reports/faq_generation_metrics.json")
    parser.add_argument("--per-sample-output", default="evaluation/reports/faq_generation_per_sample.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    base_url = args.base_url.rstrip("/")
    samples = read_jsonl(args.testset, limit=args.limit)
    print(f"Loaded {len(samples)} test samples (limit={args.limit}).")
    print(f"Base URL: {base_url}")
    print()

    per_sample: list[dict] = []
    correctness_scores: list[int] = []
    faithfulness_scores: list[int] = []
    latencies: list[float] = []
    success_count = 0
    source_match_count = 0
    pass_count = 0
    hallucination_count = 0

    for i, sample in enumerate(samples):
        question = sample["question"]
        reference_answer = sample["reference_answer"]
        expected_title = sample["expected_procedure_title"]

        # Call /chat API
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={"question": question, "session_id": f"eval-faq-{i}"},
                timeout=180,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            if response.status_code != 200:
                per_sample.append({
                    "question": question,
                    "status": "error",
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 1),
                })
                continue

            success_count += 1
            data = response.json()
            model_answer = data.get("answer", "")
            sources = data.get("sources", [])

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            per_sample.append({
                "question": question,
                "status": "error",
                "error": str(exc),
                "latency_ms": round(latency_ms, 1),
            })
            continue

        # Check source title match
        src_match = _source_title_match(sources, expected_title)
        if src_match:
            source_match_count += 1

        # LLM-as-judge
        judge_result = _call_gemini_judge(question, reference_answer, model_answer)

        result_row = {
            "question": question,
            "expected_procedure_title": expected_title,
            "status": "ok",
            "status_code": 200,
            "source_title_match": src_match,
            "latency_ms": round(latency_ms, 1),
            "model_answer_length": len(model_answer),
        }

        if judge_result:
            c_score = judge_result.get("correctness_score", 0)
            f_score = judge_result.get("faithfulness_score", 0)
            verdict = judge_result.get("final_verdict", "fail")
            hallucinated = judge_result.get("hallucinated_information", [])

            correctness_scores.append(c_score)
            faithfulness_scores.append(f_score)
            if verdict == "pass":
                pass_count += 1
            if hallucinated:
                hallucination_count += 1

            result_row.update({
                "correctness_score": c_score,
                "faithfulness_score": f_score,
                "missing_information": judge_result.get("missing_information", []),
                "hallucinated_information": hallucinated,
                "final_verdict": verdict,
            })
        else:
            result_row["judge_error"] = "Failed to get judge response"

        per_sample.append(result_row)
        print(f"  [{i+1}/{len(samples)}] {'PASS' if judge_result and judge_result.get('final_verdict') == 'pass' else 'FAIL'} "
              f"src_match={src_match} latency={round(latency_ms)}ms")

        # Brief pause between judge calls to avoid rate limiting
        # 4.5 seconds ensures we stay well under the 15 requests per minute free tier limit
        time.sleep(4.5)

    # Aggregate metrics
    n = len(samples)
    judged = len(correctness_scores)
    sorted_latencies = sorted(latencies) if latencies else [0]

    metrics = {
        "count": n,
        "success_rate": round(success_count / max(1, n), 4),
        "source_title_match_rate": round(source_match_count / max(1, success_count), 4),
        "judged_count": judged,
        "answer_correctness_avg": round(statistics.mean(correctness_scores), 2) if correctness_scores else 0,
        "faithfulness_avg": round(statistics.mean(faithfulness_scores), 2) if faithfulness_scores else 0,
        "pass_rate": round(pass_count / max(1, judged), 4),
        "hallucination_rate": round(hallucination_count / max(1, judged), 4),
        "latency_p50_ms": round(sorted_latencies[len(sorted_latencies) // 2], 1),
        "latency_p95_ms": round(sorted_latencies[int(0.95 * (len(sorted_latencies) - 1))], 1),
    }

    write_json(args.output, metrics)
    write_jsonl(args.per_sample_output, per_sample)

    print(f"\n{'='*50}")
    print("Generation Evaluation Results")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nSaved metrics to {args.output}")
    print(f"Saved per-sample results to {args.per_sample_output}")


if __name__ == "__main__":
    main()
