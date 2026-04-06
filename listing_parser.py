"""
Pure parsing helpers for property-search responses.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PropertyListing:
    name: str
    price: str
    rating: str
    rating_num: float
    desc: str
    link: str

    @property
    def cache_key(self) -> str:
        stable_id = f"{self.link}|{self.name}|{self.price}|{self.rating}"
        return hashlib.md5(stable_id.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParsedAssistantResponse:
    listings: list[PropertyListing]
    tips: list[str]
    notes: list[str]

    @property
    def has_structured_content(self) -> bool:
        return bool(self.listings or self.tips)


def extract_url(text: str) -> tuple[str, str]:
    """Extract a URL from raw text or markdown links."""
    markdown_match = re.search(r"\[([^\]]*)\]\((https?://[^\)]+)\)", text)
    if markdown_match:
        return markdown_match.group(1).strip() or text, markdown_match.group(2).strip()

    bare_match = re.search(r"(https?://\S+)", text)
    if bare_match:
        url = bare_match.group(1).rstrip(")")
        clean_text = text.replace(url, "").strip().strip("|").strip()
        return clean_text or text, url

    return text, ""


def is_separator(line: str) -> bool:
    return bool(re.match(r"^\|?[\s\-:]+(\|[\s\-:]+)+\|?$", line))


def is_header(line: str) -> bool:
    keywords = ("name", "price", "rating", "description", "link", "url", "property")
    lower = line.lower()
    matches = sum(1 for keyword in keywords if keyword in lower)
    return matches >= 2


def parse_row(line: str) -> PropertyListing | None:
    """Parse a pipe-delimited listing row into a structured object."""
    parts = [part.strip() for part in line.split("|")]
    parts = [part for part in parts if part]

    if len(parts) < 2:
        return None

    extracted_url = ""
    cleaned_parts: list[str] = []
    for part in parts:
        cleaned_text, found_url = extract_url(part)
        if found_url and not extracted_url:
            extracted_url = found_url
        if cleaned_text:
            cleaned_parts.append(cleaned_text)

    parts = [part for part in cleaned_parts if not re.match(r"^https?://", part.strip())]
    while len(parts) < 4:
        parts.append("")

    name = parts[0].strip()
    price = parts[1].strip() if len(parts) > 1 else ""
    rating = parts[2].strip() if len(parts) > 2 else ""
    desc = " ".join(parts[3:]).strip() if len(parts) > 3 else ""

    desc, extra_url = extract_url(desc)
    if extra_url and not extracted_url:
        extracted_url = extra_url

    if not name or not any((price, rating, extracted_url)):
        return None

    rating_clean = re.sub(r"[^\d.]", "", rating)
    try:
        rating_num = float(rating_clean)
    except ValueError:
        rating_num = 0.0

    return PropertyListing(
        name=name,
        price=price or "N/A",
        rating=rating_clean or rating or "N/A",
        rating_num=rating_num,
        desc=desc or "Not enough descriptive detail provided.",
        link=extracted_url,
    )


def parse_assistant_response(response: str) -> ParsedAssistantResponse:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    listings: list[PropertyListing] = []
    notes: list[str] = []
    tips: list[str] = []
    seen_listing_ids: set[str] = set()

    for line in lines:
        if is_separator(line) or is_header(line):
            continue

        if "|" in line:
            listing = parse_row(line)
            if not listing:
                continue
            dedupe_key = listing.link or f"{listing.name}|{listing.price}"
            if dedupe_key in seen_listing_ids:
                continue
            seen_listing_ids.add(dedupe_key)
            listings.append(listing)
            continue

        if re.match(r"^\d+[\.\)]\s", line):
            tips.append(re.sub(r"^\d+[\.\)]\s*", "", line).strip())
            continue

        if re.match(r"^[-*•]\s", line):
            tips.append(re.sub(r"^[-*•]\s*", "", line).strip())
            continue

        notes.append(line)

    return ParsedAssistantResponse(listings=listings, tips=tips, notes=notes)


def rating_color(value: float) -> str:
    if value >= 4.8:
        return "#2c7a6b"
    if value >= 4.4:
        return "#c48a3a"
    return "#b45343"
