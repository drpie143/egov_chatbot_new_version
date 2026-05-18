"""Run the full FAQ benchmark pipeline: retrieval ablation, generation judge, and latency.

Usage:
    python evaluation/run_faq_benchmark.py \
        --testset evaluation/testsets/dvc_faq_qa_500.jsonl \
        --base-url http://localhost:7860 \
        --generation-limit 100
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "evaluation" / "reports"


def run(command: list[str]) -> tuple[bool, str]:
    """Run a subprocess and return (success, output)."""
    print(f"  $ {' '.join(command)}")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        print(f"    FAILED (exit code {completed.returncode})")
        if output:
            # Print last 500 chars of output for debugging
            print(f"    {output[-500:]}")
    return completed.returncode == 0, output


def api_available(base_url: str) -> bool:
    """Check if the API server is running."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def read_json(path: Path) -> dict:
    """Read a JSON file, returning {} if not found."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the full FAQ benchmark pipeline.")
    parser.add_argument("--testset", default="evaluation/testsets/dvc_faq_qa_500.jsonl")
    parser.add_argument("--base-url", default="http://localhost:7860")
    parser.add_argument("--generation-limit", type=int, default=100)
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip retrieval evaluation.")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation evaluation.")
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency evaluation.")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    testset = args.testset
    base_url = args.base_url
    steps: list[dict] = []

    print("=" * 60)
    print("eGov-Bot FAQ Benchmark Runner")
    print(f"Testset: {testset}")
    print(f"Base URL: {base_url}")
    print(f"Time: {datetime.datetime.now().isoformat()}")
    print("=" * 60)

    # Step 1: Validate testset
    print("\n[Step 1] Validating testset...")
    ok, output = run([sys.executable, "evaluation/validate_faq_testset.py", "--testset", testset])
    steps.append({"name": "validate_testset", "ok": ok, "output": output[-500:]})
    if not ok:
        print("Testset validation failed. Aborting.")
        raise SystemExit(1)

    # Step 2-4: Retrieval ablation (BM25, Dense, Hybrid)
    retrieval_results: dict[str, dict] = {}
    if not args.skip_retrieval:
        for mode in ["bm25", "dense", "hybrid"]:
            print(f"\n[Step] Retrieval evaluation (mode={mode})...")
            output_file = f"evaluation/reports/faq_retrieval_{mode}.json"
            per_sample_file = f"evaluation/reports/faq_retrieval_{mode}_per_sample.jsonl"
            ok, output = run([
                sys.executable, "evaluation/eval_retrieval_title.py",
                "--testset", testset,
                "--mode", mode,
                "--output", output_file,
                "--per-sample-output", per_sample_file,
            ])
            steps.append({"name": f"retrieval_{mode}", "ok": ok, "output": output[-500:]})
            if ok:
                retrieval_results[mode] = read_json(Path(output_file))
    else:
        steps.append({"name": "retrieval", "ok": False, "output": "Skipped by user."})

    # Step 5: Check API availability
    has_api = api_available(base_url)
    print(f"\n[Step] API health check: {'available' if has_api else 'not available'}")

    # Step 6: Generation judge
    if not args.skip_generation and has_api:
        print(f"\n[Step] Generation evaluation (limit={args.generation_limit})...")
        ok, output = run([
            sys.executable, "evaluation/eval_generation_judge.py",
            "--base-url", base_url,
            "--testset", testset,
            "--limit", str(args.generation_limit),
            "--output", "evaluation/reports/faq_generation_metrics.json",
            "--per-sample-output", "evaluation/reports/faq_generation_per_sample.jsonl",
        ])
        steps.append({"name": "generation_judge", "ok": ok, "output": output[-500:]})
    else:
        reason = "Skipped by user." if args.skip_generation else "API not available."
        steps.append({"name": "generation_judge", "ok": False, "output": reason})

    # Step 7: Latency benchmark
    if not args.skip_latency and has_api:
        print("\n[Step] Latency benchmark...")
        ok, output = run([
            sys.executable, "evaluation/eval_latency_dataset.py",
            "--base-url", base_url,
            "--testset", testset,
            "--limit", "100",
            "--output", "evaluation/reports/faq_latency_metrics.json",
        ])
        steps.append({"name": "latency", "ok": ok, "output": output[-500:]})
    else:
        reason = "Skipped by user." if args.skip_latency else "API not available."
        steps.append({"name": "latency", "ok": False, "output": reason})

    # Step 8: Write latest_metrics.json
    latest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "testset": testset,
        "retrieval": retrieval_results,
        "generation": read_json(REPORT_DIR / "faq_generation_metrics.json"),
        "latency": read_json(REPORT_DIR / "faq_latency_metrics.json"),
        "steps": [{k: v for k, v in s.items() if k != "output"} for s in steps],
    }
    (REPORT_DIR / "faq_latest_metrics.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Step 9: Write latest_report.md
    report_lines = _generate_report(latest, retrieval_results)
    (REPORT_DIR / "faq_latest_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
    for step in steps:
        status = "✓" if step["ok"] else "✗"
        print(f"  {status} {step['name']}")
    print(f"\nReports written to {REPORT_DIR}/")


def _generate_report(latest: dict, retrieval_results: dict) -> list[str]:
    """Generate the benchmark report in Markdown."""
    lines = [
        "# eGov-Bot FAQ Benchmark Report",
        "",
        f"Generated: {latest['timestamp']}",
        "",
        "## 1. Testset",
        "",
        "- Source: official FAQ-style questions from Vietnamese National Public Service Portal.",
        "- Final schema: `question`, `reference_answer`, `expected_procedure_title`.",
        f"- Testset path: `{latest['testset']}`",
        "- Note: FAQ answers are used as reference answers, not as retrieval corpus.",
        "",
        "## 2. Retrieval Results",
        "",
    ]

    if retrieval_results:
        lines.append("| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p95 latency |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for mode in ["bm25", "dense", "hybrid"]:
            m = retrieval_results.get(mode, {})
            if m:
                lines.append(
                    f"| {mode} | {m.get('recall@1', '-')} | {m.get('recall@3', '-')} | "
                    f"{m.get('recall@5', '-')} | {m.get('recall@10', '-')} | "
                    f"{m.get('mrr@10', '-')} | {m.get('ndcg@10', '-')} | "
                    f"{m.get('latency_p95_ms', '-')} ms |"
                )
        lines.append("")
    else:
        lines.append("_Retrieval evaluation was not run._")
        lines.append("")

    lines.extend([
        "## 3. Generation Results",
        "",
    ])

    gen = latest.get("generation", {})
    if gen:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for key in [
            "success_rate", "source_title_match_rate", "answer_correctness_avg",
            "faithfulness_avg", "pass_rate", "hallucination_rate",
            "latency_p50_ms", "latency_p95_ms",
        ]:
            val = gen.get(key, "-")
            lines.append(f"| {key} | {val} |")
        lines.append("")
    else:
        lines.append("_Generation evaluation was not run._")
        lines.append("")

    lines.extend([
        "## 4. Latency Results",
        "",
    ])

    lat = latest.get("latency", {})
    if lat:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for key in ["count", "latency_mean_ms", "latency_p50_ms", "latency_p90_ms", "latency_p95_ms", "latency_p99_ms", "latency_max_ms"]:
            val = lat.get(key, "-")
            lines.append(f"| {key} | {val} |")
        lines.append("")
    else:
        lines.append("_Latency evaluation was not run._")
        lines.append("")

    lines.extend([
        "## 5. Steps",
        "",
    ])
    for step in latest.get("steps", []):
        status = "✓" if step.get("ok") else "✗"
        lines.append(f"- {status} {step['name']}")

    lines.extend([
        "",
        "## 6. Next Improvements",
        "",
        "- Add reranker.",
        "- Add query rewriting for vague FAQ questions.",
        "- Add better citation/title matching.",
        "- Add human spot-check set.",
        "",
    ])

    return lines


if __name__ == "__main__":
    main()
