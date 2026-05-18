from egov_bot.rag.citation import extract_urls, unique_sources
from egov_bot.schemas.common import Source


def test_extract_urls_deduplicates():
    text = "Nguồn https://example.com/a và [link](https://example.com/a)."
    assert extract_urls(text) == ["https://example.com/a"]


def test_unique_sources_deduplicates_by_url():
    sources = [
        Source(title="A", url="u1"),
        Source(title="A again", url="u1"),
        Source(title="B", url="u2"),
    ]
    assert [source.url for source in unique_sources(sources)] == ["u1", "u2"]

