"""
RAG pipeline for Airbnb listing summaries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 15
MAX_RAW_CONTEXT_CHARS = 3200
MAX_SUMMARY_WORDS = 70
SCRAPE_MIN_TEXT_LENGTH = 180
RETRIEVER_QUERY = (
    "property type bedrooms bathrooms sleeps neighborhood transit amenities "
    "kitchen workspace outdoor space unique features house rules"
)

_index_cache: dict[str, FAISS] = {}
_summary_cache: dict[str, "SummaryResult"] = {}


@dataclass(frozen=True)
class SummaryResult:
    text: str
    source: str


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def _summary_cache_key(url: str, fallback_metadata: dict[str, Any] | None) -> str:
    metadata_bits = fallback_metadata or {}
    stable_payload = json.dumps(
        {
            "url": _normalize_url(url),
            "metadata": {
                "name": metadata_bits.get("name", ""),
                "price": metadata_bits.get("price", ""),
                "rating": metadata_bits.get("rating", ""),
                "desc": metadata_bits.get("desc", ""),
            },
        },
        sort_keys=True,
    )
    return hashlib.md5(stable_payload.encode("utf-8")).hexdigest()


def _validate_listing_url(url: str) -> None:
    parsed = urlparse(_normalize_url(url))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("Listing URL is missing or invalid.")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _truncate_words(text: str, max_words: int = MAX_SUMMARY_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",.;:") + "."


def _extract_metadata_text(soup: BeautifulSoup) -> list[str]:
    candidates: list[str] = []

    title = soup.title.string if soup.title and soup.title.string else ""
    meta_fields = (
        ("meta", {"property": "og:title"}, "content"),
        ("meta", {"property": "og:description"}, "content"),
        ("meta", {"name": "description"}, "content"),
        ("meta", {"name": "twitter:description"}, "content"),
    )

    if title:
        candidates.append(_clean_text(title))

    for tag_name, attrs, field_name in meta_fields:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get(field_name):
            candidates.append(_clean_text(tag.get(field_name)))

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw_json = script.string or script.get_text(" ", strip=True)
        if not raw_json:
            continue
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue

        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("name", "description", "amenityFeature"):
                value = item.get(key)
                if isinstance(value, str):
                    candidates.append(_clean_text(value))
                elif isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict):
                            entry_name = entry.get("name")
                            if isinstance(entry_name, str):
                                candidates.append(_clean_text(entry_name))
                        elif isinstance(entry, str):
                            candidates.append(_clean_text(entry))

    return [candidate for candidate in candidates if len(candidate) > 20]


def _extract_body_text(soup: BeautifulSoup) -> list[str]:
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
        tag.decompose()

    selectors = ("main", "body", "[role='main']", "section", ".page-container")
    fragments: list[str] = []

    for selector in selectors:
        container = soup.select_one(selector)
        if not container:
            continue
        for element in container.find_all(["p", "h1", "h2", "h3", "li", "div", "span"]):
            text = _clean_text(element.get_text(separator=" ", strip=True))
            if len(text) >= 25:
                fragments.append(text)

    seen: set[str] = set()
    cleaned_fragments: list[str] = []
    for fragment in fragments:
        lowered = fragment.lower()
        if lowered in seen:
            continue
        if any(marker in lowered for marker in ("airbnb your home", "skip to content")):
            continue
        seen.add(lowered)
        cleaned_fragments.append(fragment)
    return cleaned_fragments


def _scrape_listing_text(url: str) -> str:
    _validate_listing_url(url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }

    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not fetch listing page: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")
    content_blocks = _extract_metadata_text(soup) + _extract_body_text(soup)
    content_blocks = [block for block in content_blocks if block]

    if not content_blocks:
        raise RuntimeError("No extractable listing text was returned.")

    joined_text = "\n".join(content_blocks[:120])
    if len(joined_text) < SCRAPE_MIN_TEXT_LENGTH:
        raise RuntimeError("Listing page did not return enough usable text.")

    blocked_markers = ("captcha", "access denied", "verify you are human")
    if any(marker in joined_text.lower() for marker in blocked_markers):
        raise RuntimeError("Listing page appears to be blocked or protected.")

    return joined_text


@lru_cache(maxsize=4)
def _get_embeddings(api_key: str) -> Any:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)


@lru_cache(maxsize=4)
def _get_summary_llm(api_key: str) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)


def _get_or_build_index(url: str, api_key: str) -> FAISS:
    cache_key = hashlib.md5(_normalize_url(url).encode("utf-8")).hexdigest()
    if cache_key in _index_cache:
        return _index_cache[cache_key]

    if not api_key:
        raise RuntimeError("OpenAI API key is required to build a vector index.")

    listing_text = _scrape_listing_text(url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    docs = splitter.create_documents([listing_text])
    if not docs:
        raise RuntimeError("Unable to build chunks from listing text.")

    index = FAISS.from_documents(docs, _get_embeddings(api_key))
    _index_cache[cache_key] = index
    return index


def _infer_property_type(name: str, desc: str) -> str:
    text = f"{name} {desc}".lower()
    keywords = {
        "villa": "villa",
        "cabin": "cabin",
        "studio": "studio",
        "loft": "loft",
        "apartment": "apartment",
        "condo": "condo",
        "house": "house",
        "room": "private room",
        "beach": "beach stay",
    }
    for keyword, label in keywords.items():
        if keyword in text:
            return label
    return "stay"


def build_metadata_summary(fallback_metadata: dict[str, Any] | None = None) -> str:
    metadata = fallback_metadata or {}
    name = _clean_text(str(metadata.get("name", "")))
    price = _clean_text(str(metadata.get("price", "")))
    rating = _clean_text(str(metadata.get("rating", "")))
    desc = _clean_text(str(metadata.get("desc", "")))

    details: list[str] = []
    property_type = _infer_property_type(name, desc)

    if name:
        details.append(f"{name} looks like a {property_type}.")
    else:
        details.append(f"This listing looks like a {property_type}.")

    if price and price != "N/A":
        details.append(f"Listed around {price}.")

    if rating and rating not in {"N/A", "0"}:
        details.append(f"Guest rating is {rating}/5.")

    if desc and desc not in {"N/A", "Not enough descriptive detail provided."}:
        details.append(desc.rstrip(".") + ".")
    else:
        details.append("The source response includes only limited listing detail.")

    summary = _truncate_words(" ".join(details))
    return summary or "Listing summary unavailable."


def _summarize_with_rag(url: str, api_key: str) -> SummaryResult:
    index = _get_or_build_index(url, api_key)
    retriever = index.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(RETRIEVER_QUERY)
    if not docs:
        raise RuntimeError("Retriever returned no useful listing context.")

    context = "\n\n".join(doc.page_content for doc in docs if getattr(doc, "page_content", ""))
    if not context.strip():
        raise RuntimeError("Retrieved listing context was empty.")

    prompt = (
        "Summarize this Airbnb listing in 70 words or fewer. "
        "Cover the property type, likely room layout, location cues, two or three notable amenities, "
        "and one concrete standout detail. Stay factual. Do not use bullets or marketing language.\n\n"
        f"Listing content:\n{context[:MAX_RAW_CONTEXT_CHARS]}"
    )
    response = _get_summary_llm(api_key).invoke(prompt)
    content = getattr(response, "content", "") or ""
    cleaned_summary = _truncate_words(_clean_text(content))
    if not cleaned_summary:
        raise RuntimeError("LLM returned an empty summary.")

    return SummaryResult(text=cleaned_summary, source="retrieved")


def summarize_listing(
    url: str,
    api_key: str,
    fallback_metadata: dict[str, Any] | None = None,
) -> SummaryResult:
    """
    Return a robust listing summary.

    The function prefers retrieved listing content and falls back to metadata-driven
    summarization if scraping, embedding, retrieval, or generation fails.
    """

    cache_key = _summary_cache_key(url, fallback_metadata)
    if cache_key in _summary_cache:
        return _summary_cache[cache_key]

    fallback_result = SummaryResult(
        text=build_metadata_summary(fallback_metadata),
        source="metadata",
    )

    if not url:
        _summary_cache[cache_key] = fallback_result
        return fallback_result

    try:
        _validate_listing_url(url)
    except RuntimeError:
        _summary_cache[cache_key] = fallback_result
        return fallback_result

    if not api_key:
        _summary_cache[cache_key] = fallback_result
        return fallback_result

    try:
        summary = _summarize_with_rag(url, api_key)
    except Exception as exc:
        logger.info("Falling back to metadata summary for %s: %s", url, exc)
        summary = fallback_result

    _summary_cache[cache_key] = summary
    return summary
