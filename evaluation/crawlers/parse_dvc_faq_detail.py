"""Parse a FAQ detail page from dichvucong.gov.vn.

Extracts question, reference_answer, and expected_procedure_title from the HTML.

Usage:
    python evaluation/crawlers/parse_dvc_faq_detail.py \
        --url "https://dichvucong.gov.vn/p/home/dvc-chi-tiet-cau-hoi.html?id=15180&row_limit=1"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.utils.text_normalize import normalize_text  # noqa: E402

USER_AGENT = "Mozilla/5.0 (compatible; eGov-Bot academic benchmark crawler)"

# Minimum lengths for valid samples
MIN_QUESTION_LEN = 15
MIN_ANSWER_LEN = 30
MIN_TITLE_LEN = 10

# Answer patterns to reject
REJECT_PATTERNS = [
    "đang cập nhật",
    "đang xử lý",
    "không có thông tin",
    "vui lòng xem thủ tục liên quan",
]


def _get_clean_lines(text: str) -> list[str]:
    """Split text into non-empty stripped lines."""
    return [line.strip() for line in text.split("\n") if line.strip()]


def _extract_between(
    lines: list[str],
    start_marker: str,
    end_markers: list[str],
) -> str | None:
    """Extract text between start_marker and any of the end_markers."""
    start_idx = None
    for i, line in enumerate(lines):
        if start_marker.lower() in line.lower():
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        for marker in end_markers:
            if marker.lower() in lines[i].lower():
                end_idx = i
                break
        if end_idx != len(lines):
            break

    content_lines = lines[start_idx:end_idx]
    if not content_lines:
        return None
    return " ".join(content_lines)


def _extract_after_marker(
    lines: list[str],
    marker: str,
    stop_markers: list[str],
) -> str | None:
    """Extract the first meaningful text after marker, before any stop marker."""
    start_idx = None
    for i, line in enumerate(lines):
        if marker.lower() in line.lower():
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        # Stop if we hit a stop marker
        for stop in stop_markers:
            if stop.lower() in line.lower():
                return None
        # Skip navigation/UI text
        if line in ("Xem thêm", "Thu nhỏ", "Tìm kiếm"):
            continue
        return line

    return None


def parse_detail(html_content: str) -> dict | None:
    """Parse a FAQ detail page HTML and return structured data.

    Returns dict with question, reference_answer, expected_procedure_title
    or None if parsing fails or sample is invalid.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Try to get question from h1
    question = None
    h1 = soup.find("h1")
    if h1:
        question = normalize_text(h1.get_text(strip=True))

    # Get full text and clean lines for fallback parsing
    full_text = soup.get_text("\n")
    lines = _get_clean_lines(full_text)

    # Extract answer: text after "Trả lời:" until "Các thủ tục liên quan"
    answer = _extract_between(
        lines,
        start_marker="Trả lời:",
        end_markers=["Các thủ tục liên quan", "Các câu hỏi liên quan"],
    )

    # Extract expected procedure title: first item after "Các thủ tục liên quan"
    expected_title = None

    # Try structured parsing first: look for links in the related procedure section
    related_section = None
    for tag in soup.find_all(["h2", "h3", "div", "span"]):
        if "thủ tục liên quan" in (tag.get_text(strip=True) or "").lower():
            related_section = tag
            break

    if related_section:
        # Look for the first link after this section
        next_elem = related_section.find_next("a")
        if next_elem:
            title_candidate = normalize_text(next_elem.get_text(strip=True))
            if len(title_candidate) >= MIN_TITLE_LEN:
                expected_title = title_candidate

    # Fallback: use text-based extraction
    if not expected_title:
        expected_title = _extract_after_marker(
            lines,
            marker="Các thủ tục liên quan",
            stop_markers=["Các câu hỏi liên quan", "Thu nhỏ câu hỏi liên quan"],
        )

    if expected_title:
        expected_title = normalize_text(expected_title)

    # Validate sample
    if not question or len(question) < MIN_QUESTION_LEN:
        return None
    if not answer or len(answer) < MIN_ANSWER_LEN:
        return None
    if not expected_title or len(expected_title) < MIN_TITLE_LEN:
        return None

    # Check for rejected patterns
    answer_lower = answer.lower()
    question_lower = question.lower()
    for pattern in REJECT_PATTERNS:
        if pattern in answer_lower or pattern in question_lower:
            return None

    return {
        "question": question,
        "reference_answer": normalize_text(answer),
        "expected_procedure_title": expected_title,
    }


def fetch_and_parse(url: str, timeout: int = 20) -> dict | None:
    """Fetch a URL and parse the FAQ detail page."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        return parse_detail(response.text)
    except Exception as exc:
        print(f"  Error fetching {url}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description="Parse a single FAQ detail page.")
    parser.add_argument(
        "--url",
        required=True,
        help="URL of the FAQ detail page to parse.",
    )
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    result = fetch_and_parse(args.url, timeout=args.timeout)
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Failed to parse or invalid sample.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
