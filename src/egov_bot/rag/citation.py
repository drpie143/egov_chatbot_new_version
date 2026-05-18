from __future__ import annotations

import re

from egov_bot.schemas.common import Source

URL_RE = re.compile(r"https?://[^\s)\]}\"']+")


def extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in URL_RE.findall(text or ""):
        url = match.rstrip(".,;")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def unique_sources(sources: list[Source], limit: int = 5) -> list[Source]:
    seen: set[str] = set()
    unique: list[Source] = []
    for source in sources:
        key = source.url or source.title
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(source)
        if len(unique) >= limit:
            break
    return unique


def append_sources_if_missing(answer: str, sources: list[Source]) -> str:
    if not sources:
        return answer
    answer_urls = set(extract_urls(answer))
    missing = [source for source in sources if source.url and source.url not in answer_urls]
    if not missing:
        return answer
    lines = ["", "Nguồn tham khảo:"]
    for index, source in enumerate(missing[:3], start=1):
        lines.append(f"{index}. {source.title}: {source.url}")
    return answer.rstrip() + "\n".join(lines)

