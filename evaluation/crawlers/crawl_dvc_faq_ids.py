"""Crawl FAQ detail pages from dichvucong.gov.vn by scanning an ID range.

Usage:
    python evaluation/crawlers/crawl_dvc_faq_ids.py \
        --start-id 1 \
        --end-id 30000 \
        --max-valid 800 \
        --output evaluation/testsets/dvc_faq_raw.jsonl \
        --sleep-min 0.7 \
        --sleep-max 2.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.crawlers.parse_dvc_faq_detail import fetch_and_parse  # noqa: E402

FAQ_DETAIL_URL = "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-cau-hoi.html?id={id}&row_limit=1"


def main() -> None:
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description="Crawl FAQ pages by scanning ID range.")
    parser.add_argument("--start-id", type=int, default=1, help="Start of ID range to scan.")
    parser.add_argument("--end-id", type=int, default=30000, help="End of ID range to scan.")
    parser.add_argument("--max-valid", type=int, default=800, help="Stop after collecting this many valid samples.")
    parser.add_argument("--output", default="evaluation/testsets/dvc_faq_raw.jsonl", help="Output JSONL path.")
    parser.add_argument("--sleep-min", type=float, default=0.7, help="Min sleep between requests (seconds).")
    parser.add_argument("--sleep-max", type=float, default=2.0, help="Max sleep between requests (seconds).")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP request timeout (seconds).")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request.")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N valid samples.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    existing: set[int] = set()
    results: list[dict] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    results.append(row)
                    if "faq_id" in row:
                        existing.add(row["faq_id"])
        print(f"Resumed: {len(results)} existing samples loaded.")

    valid_count = len(results)
    scanned = 0
    errors = 0

    print(f"Scanning IDs {args.start_id}..{args.end_id}, target: {args.max_valid} valid samples")
    print(f"Sleep: {args.sleep_min}-{args.sleep_max}s, timeout: {args.timeout}s")
    print()

    try:
        for faq_id in range(args.start_id, args.end_id + 1):
            if valid_count >= args.max_valid:
                print(f"\nReached target: {valid_count} valid samples.")
                break

            if faq_id in existing:
                continue

            url = FAQ_DETAIL_URL.format(id=faq_id)
            scanned += 1

            # Retry logic
            result = None
            for attempt in range(1, args.max_retries + 1):
                result = fetch_and_parse(url, timeout=args.timeout)
                if result is not None:
                    break
                if attempt < args.max_retries:
                    backoff = args.sleep_max * (2 ** (attempt - 1))
                    time.sleep(backoff)

            if result is not None:
                result["faq_id"] = faq_id
                result["url"] = url
                result["crawl_status"] = "ok"
                results.append(result)
                valid_count += 1
                print(f"  [{valid_count}/{args.max_valid}] ID={faq_id} ✓ {result['question'][:60]}...")

                # Periodic checkpoint
                if valid_count % args.checkpoint_every == 0:
                    _save(output_path, results)
                    print(f"  Checkpoint saved: {valid_count} samples")
            else:
                errors += 1

            # Rate limiting
            sleep_time = random.uniform(args.sleep_min, args.sleep_max)
            time.sleep(sleep_time)

            # Progress update every 100 scanned
            if scanned % 100 == 0:
                print(f"  Scanned {scanned} IDs, found {valid_count} valid, {errors} errors")

    except KeyboardInterrupt:
        print(f"\nInterrupted. Saving {valid_count} samples...")

    _save(output_path, results)
    print(f"\nDone. Scanned {scanned} IDs, saved {valid_count} valid samples to {output_path}")
    print(f"Errors/empty: {errors}")


def _save(path: Path, results: list[dict]) -> None:
    """Write all results to JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
