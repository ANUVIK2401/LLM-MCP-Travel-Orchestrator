"""
Streamlit travel assistant for live Airbnb search and property findings.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from shutil import which
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import nest_asyncio
import rag
from listing_parser import ParsedAssistantResponse, PropertyListing, parse_assistant_response, rating_color
from mcp_use import MCPAgent, MCPClient

nest_asyncio.apply()

# Python 3.13 compatibility fix for asyncio + pydantic deepcopy issues.
_original_deepcopy = copy.deepcopy


def _safe_deepcopy(x, memo=None, _nil=[]):
    try:
        return _original_deepcopy(x, memo, _nil)
    except (TypeError, AttributeError) as exc:
        if any(keyword in str(exc) for keyword in ("pickle", "Future", "_asyncio", "Context")):
            return x
        raise


copy.deepcopy = _safe_deepcopy

try:
    import pydantic.v1.utils as pyd_utils

    _original_smart_deepcopy = pyd_utils.smart_deepcopy

    def _safe_smart_deepcopy(obj):
        try:
            return _original_smart_deepcopy(obj)
        except (TypeError, AttributeError) as exc:
            if any(keyword in str(exc) for keyword in ("pickle", "Future", "_asyncio", "Context")):
                return obj
            raise

    pyd_utils.smart_deepcopy = _safe_smart_deepcopy
except ImportError:
    pass

st.set_page_config(
    page_title="Travel Property Findings",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
MCP_CONFIG_PATH = APP_DIR / "airbnb_mcp.json"


def _get_default_dates() -> tuple[str, str]:
    checkin = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    checkout = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
    return checkin, checkout


_CHECKIN, _CHECKOUT = _get_default_dates()

CUSTOM_SYSTEM_PROMPT = f"""
You are a travel assistant with access to these tools:

{{{{tool_descriptions}}}}

When users ask about accommodations, properties, or hotels in any city, ALWAYS call the airbnb_search tool.

== IMPORTANT: HOW PRICING WORKS ==
The airbnb_search tool's minPrice and maxPrice represent the TOTAL price for the ENTIRE stay (all nights combined).
The default stay is 1 night (checkin="{_CHECKIN}", checkout="{_CHECKOUT}").
With a 1-night stay, maxPrice directly equals the per-night budget.
Example: User says "under $100/night" -> set maxPrice=100 (1-night stay = $100 total = $100/night).
If the user specifies a different number of nights N, multiply: maxPrice = budget_per_night x N.

== TOOL PARAMETERS (airbnb_search) ==
Supported: location, checkin, checkout, adults, children, minPrice, maxPrice
NOT supported: pool, amenities, gym, wifi, or any other feature/amenity filters

== RULES ==
1. ALWAYS include checkin="{_CHECKIN}" and checkout="{_CHECKOUT}" unless the user specifies dates.
2. ALWAYS apply minPrice/maxPrice when the user mentions a budget or price limit.
3. If the user asks for amenities (pool, gym, wifi, etc.) then search by location only,
   then note in your reply that amenity filtering is not supported by the search tool.
4. If a price-filtered search returns 0 results, DO NOT silently remove the filter.
   Instead, tell the user that no listings were found within budget and retry without the price filter.
5. Return results as a pipe-delimited table with EXACTLY 5 columns per row (no header, no separator):
   Name | Price | Rating | Description | URL
6. URL must be the raw Airbnb URL (https://www.airbnb.com/rooms/...) with no markdown link syntax.
7. Price must be formatted as per-night (for example $97/night) even if the tool returns a total price.
8. Only say "No listings found" if retrying without any filters also returns 0 results.
"""

QUICK_SEARCHES = [
    "Affordable Airbnbs in Paris under $150 per night",
    "Family-friendly stays in Bali under $180",
    "Apartments near Central Park in New York City",
    "Budget stays in Tokyo under $90 per night",
    "Beachfront rentals in Barcelona for 2 adults",
]

APP_CSS = """
<style>
    :root {
        --bg: #080808;
        --bg-alt: #111111;
        --panel: #141414;
        --panel-soft: #1d1d1d;
        --panel-strong: #1a1a1a;
        --panel-glass: rgba(20, 20, 20, 0.92);
        --ink: #f5f5f1;
        --muted: #b9b9b2;
        --line: #2d2d2d;
        --red: #e50914;
        --red-soft: rgba(229, 9, 20, 0.14);
        --red-strong: #ff3b30;
        --amber: #f5c451;
        --green: #35d07f;
        --shadow: 0 26px 60px rgba(0, 0, 0, 0.45);
        --radius: 20px;
    }

    html, body, [class*="css"] {
        font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    }

    h1, h2, h3, .hero-title, .section-title {
        font-family: "Georgia", "Times New Roman", serif;
    }

    [data-testid="stAppViewContainer"] > .main {
        background:
            radial-gradient(circle at top center, rgba(229, 9, 20, 0.18), transparent 22%),
            radial-gradient(circle at 15% 25%, rgba(255, 255, 255, 0.03), transparent 18%),
            linear-gradient(180deg, #1a0708 0%, #0f0f0f 22%, var(--bg) 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0b0b 0%, #121212 100%);
        border-right: 1px solid var(--line);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stChatInput"] textarea {
        background: rgba(18, 18, 18, 0.96) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 18px !important;
        font-size: 0.96rem !important;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #808080 !important;
    }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.35rem 0 !important;
    }

    .stButton > button {
        border-radius: 14px;
        border: 1px solid var(--line);
        background: linear-gradient(180deg, #1b1b1b 0%, #121212 100%);
        color: var(--ink);
        font-weight: 600;
        min-height: 3rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(229, 9, 20, 0.45);
        box-shadow: 0 16px 30px rgba(0, 0, 0, 0.42);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 1.4rem;
        padding-bottom: 2.2rem;
    }

    .block-container > div {
        animation: fadeUp 0.24s ease-out;
    }

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(6px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: var(--ink);
    }

    .hero-shell {
        background:
            linear-gradient(135deg, rgba(28, 10, 12, 0.98) 0%, rgba(12, 12, 12, 0.96) 52%, rgba(26, 9, 10, 0.98) 100%);
        border: 1px solid rgba(229, 9, 20, 0.18);
        border-radius: 28px;
        padding: 1.8rem 1.8rem 1.45rem 1.8rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-shell::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: linear-gradient(180deg, rgba(255, 91, 91, 0.9), rgba(229, 9, 20, 0.15));
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        inset: auto -60px -80px auto;
        width: 220px;
        height: 220px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(229, 9, 20, 0.28), transparent 68%);
    }

    .hero-kicker {
        color: #ff7b7b;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.72rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }

    .hero-title {
        margin: 0;
        color: var(--ink);
        font-size: 2.4rem;
        line-height: 1.02;
    }

    .hero-copy {
        margin: 0.8rem 0 0 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
        max-width: 760px;
    }

    .hero-eyebrow-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #e3e3dc;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .hero-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.2rem;
    }

    .hero-stat {
        min-width: 150px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 0.8rem 0.95rem;
    }

    .hero-stat-label {
        margin: 0;
        color: #949494;
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
    }

    .hero-stat-value {
        margin: 0.35rem 0 0 0;
        color: var(--ink);
        font-size: 0.98rem;
        font-weight: 700;
    }

    .status-banner {
        border-radius: 20px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .status-banner.warning {
        background: linear-gradient(135deg, rgba(63, 16, 18, 0.96), rgba(28, 16, 16, 0.96));
    }

    .status-banner.ready {
        background: linear-gradient(135deg, rgba(18, 18, 18, 0.98), rgba(25, 25, 25, 0.98));
    }

    .status-title {
        margin: 0 0 0.35rem 0;
        color: var(--ink);
        font-size: 0.98rem;
        font-weight: 700;
    }

    .status-copy {
        margin: 0;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .sidebar-card {
        background: rgba(24, 24, 24, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.22);
    }

    .sidebar-title {
        margin: 0;
        color: var(--ink);
        font-size: 1rem;
        font-weight: 700;
    }

    .sidebar-copy, .sidebar-list {
        margin: 0.4rem 0 0 0;
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.55;
    }

    .note-card, .tips-card {
        background: rgba(18, 18, 18, 0.94);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        margin-bottom: 0.9rem;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }

    .note-card {
        border-left: 4px solid var(--red);
    }

    .tips-card {
        border-left: 4px solid var(--amber);
    }

    .section-title {
        margin: 0 0 0.5rem 0;
        color: var(--ink);
        font-size: 1.35rem;
    }

    .section-copy {
        margin: 0;
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.6;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        overflow: hidden;
        box-shadow: var(--shadow);
        background: linear-gradient(180deg, rgba(20, 20, 20, 0.98) 0%, rgba(12, 12, 12, 0.98) 100%);
    }

    div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
        background: transparent;
    }

    div[data-testid="stDataFrame"] [role="columnheader"] {
        background: #111111 !important;
        color: #cfcfc6 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stDataFrame"] [role="gridcell"] {
        background: transparent !important;
        color: var(--ink) !important;
        border-color: rgba(255, 255, 255, 0.04) !important;
    }

    .table-shell {
        background: linear-gradient(180deg, rgba(22, 22, 22, 0.98) 0%, rgba(14, 14, 14, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        overflow: hidden;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .table-scroll {
        overflow-x: auto;
    }

    .listing-table {
        width: 100%;
        border-collapse: collapse;
        min-width: 900px;
    }

    .listing-table thead th {
        background: #0b0b0b;
        color: #8d8d8d;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        padding: 0.95rem 1rem;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .listing-table tbody tr {
        transition: background 0.18s ease;
    }

    .listing-table tbody tr:nth-child(even) {
        background: rgba(255, 255, 255, 0.015);
    }

    .listing-table tbody tr:hover {
        background: rgba(229, 9, 20, 0.08);
    }

    .listing-table td {
        padding: 1rem;
        color: var(--ink);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        vertical-align: top;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .table-rank {
        width: 44px;
        color: #8f8f8f;
        font-weight: 700;
    }

    .table-property-name {
        color: var(--ink);
        text-decoration: none;
        font-weight: 700;
    }

    .table-property-name:hover {
        color: #ffffff;
    }

    .table-description {
        display: block;
        margin-top: 0.35rem;
        color: var(--muted);
        font-size: 0.84rem;
    }

    .table-price {
        color: #ffffff;
        font-weight: 700;
    }

    .table-rating {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.28rem 0.62rem;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid currentColor;
    }

    .table-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 96px;
        padding: 0.5rem 0.75rem;
        border-radius: 999px;
        background: linear-gradient(180deg, #f31d28 0%, #c90511 100%);
        color: white;
        text-decoration: none;
        font-weight: 700;
        font-size: 0.8rem;
    }

    .table-link:hover {
        background: linear-gradient(180deg, #ff434c 0%, #e50914 100%);
    }

    .property-card {
        background: linear-gradient(180deg, rgba(21, 21, 21, 0.98) 0%, rgba(12, 12, 12, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 24px;
        padding: 1.15rem 1.2rem;
        margin: 0.95rem 0;
        box-shadow: var(--shadow);
    }

    .property-topline {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .property-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 999px;
        background: linear-gradient(180deg, #f31d28 0%, #8f030c 100%);
        color: #fff;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .property-name {
        color: var(--ink);
        font-size: 1.12rem;
        font-weight: 700;
        text-decoration: none;
    }

    .property-name:hover {
        color: #ffffff;
    }

    .property-description {
        margin: 0.45rem 0 0 0;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.65;
    }

    .property-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.85rem;
    }

    .property-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.38rem 0.7rem;
        background: rgba(255, 255, 255, 0.04);
        color: var(--ink);
        font-size: 0.82rem;
        font-weight: 700;
    }

    .property-summary {
        margin-top: 1rem;
        border-radius: 18px;
        padding: 0.92rem 1rem;
        border: 1px solid rgba(229, 9, 20, 0.18);
        background: linear-gradient(135deg, rgba(48, 8, 10, 0.96), rgba(20, 20, 20, 0.98));
    }

    .property-summary.metadata {
        border-color: rgba(245, 196, 81, 0.18);
        background: linear-gradient(135deg, rgba(39, 30, 8, 0.94), rgba(20, 20, 20, 0.98));
    }

    .summary-eyebrow {
        margin: 0 0 0.35rem 0;
        color: #ff8f8f;
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .summary-eyebrow.metadata {
        color: var(--amber);
    }

    .summary-body {
        margin: 0;
        color: var(--ink);
        font-size: 0.9rem;
        line-height: 1.72;
    }

    .property-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-top: 1rem;
        padding-top: 0.85rem;
        border-top: 1px dashed rgba(255, 255, 255, 0.08);
    }

    .property-link {
        color: #ff666b;
        font-size: 0.83rem;
        font-weight: 700;
        text-decoration: none;
    }

    .property-link:hover {
        color: #ffffff;
    }

    .empty-state {
        background: rgba(18, 18, 18, 0.94);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        color: var(--muted);
    }

    .summary-card {
        background: linear-gradient(180deg, rgba(24, 24, 24, 0.98) 0%, rgba(14, 14, 14, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 22px;
        padding: 1.05rem 1.15rem;
        margin: 0 0 1rem 0;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .summary-card::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 3px;
        background: linear-gradient(90deg, rgba(229, 9, 20, 0.95), rgba(255, 118, 118, 0.35));
    }

    .summary-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 24px 48px rgba(0, 0, 0, 0.32);
    }

    .summary-card.retrieved {
        border-color: rgba(229, 9, 20, 0.24);
        box-shadow: 0 18px 40px rgba(229, 9, 20, 0.08);
    }

    .summary-card.metadata {
        border-color: rgba(245, 196, 81, 0.18);
    }

    .summary-header {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 0.8rem;
        margin-bottom: 0.75rem;
    }

    .summary-title {
        margin: 0;
        color: var(--ink);
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.35;
    }

    .summary-subtitle {
        margin: 0.2rem 0 0 0;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.45;
    }

    .summary-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 999px;
        background: rgba(229, 9, 20, 0.12);
        border: 1px solid rgba(229, 9, 20, 0.28);
        color: #ffd5d5;
        font-size: 0.78rem;
        font-weight: 700;
        flex-shrink: 0;
    }

    .summary-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.7rem 0 0.85rem 0;
    }

    .summary-badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.35rem 0.62rem;
        font-size: 0.77rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.07);
        color: var(--ink);
    }

    .section-caption {
        color: #8d8d87;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .summary-badge.source {
        color: #ffadad;
        border-color: rgba(229, 9, 20, 0.25);
        background: rgba(229, 9, 20, 0.08);
    }

    .summary-badge.metadata {
        color: var(--amber);
        border-color: rgba(245, 196, 81, 0.2);
        background: rgba(245, 196, 81, 0.06);
    }

    .summary-text {
        margin: 0;
        color: var(--ink);
        font-size: 0.9rem;
        line-height: 1.72;
    }

    .summary-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-top: 0.9rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .summary-link {
        color: #76b9ff;
        font-size: 0.82rem;
        font-weight: 700;
        text-decoration: none;
    }

    .summary-link:hover {
        color: #a7d1ff;
    }

    .summary-footer-note {
        color: #8d8d87;
        font-size: 0.77rem;
    }

    .results-table-shell {
        background: linear-gradient(180deg, rgba(20, 20, 20, 0.98) 0%, rgba(12, 12, 12, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        padding: 0.8rem 0.9rem 0.25rem 0.9rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
    }

    .results-table-header {
        color: #d8d8d2;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0 0.15rem 0.65rem 0.15rem;
    }

    .results-row {
        padding: 0.55rem 0.15rem 0.75rem 0.15rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .results-name {
        color: var(--ink);
        font-size: 0.89rem;
        font-weight: 700;
        line-height: 1.35;
    }

    .results-meta {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.45;
    }

    .results-cell {
        color: var(--ink);
        font-size: 0.84rem;
        line-height: 1.45;
    }

    .results-link a {
        color: #76b9ff !important;
        font-size: 0.82rem;
        font-weight: 700;
        text-decoration: none;
    }

    .results-link a:hover {
        color: #a7d1ff !important;
        text-decoration: underline;
    }

    @media (max-width: 900px) {
        .hero-title {
            font-size: 1.95rem;
        }
    }
</style>
"""


def run_async_task(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def _load_mcp_command() -> str:
    if not MCP_CONFIG_PATH.exists():
        return ""
    try:
        with MCP_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return ""

    servers = config.get("mcpServers", {})
    for server_config in servers.values():
        command = str(server_config.get("command", "")).strip()
        if command:
            return command
    return ""


def get_runtime_diagnostics() -> dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    mcp_command = _load_mcp_command()

    if not mcp_command:
        mcp_binary_ready = False
    elif os.path.isabs(mcp_command):
        mcp_binary_ready = Path(mcp_command).exists()
    else:
        mcp_binary_ready = which(mcp_command) is not None

    issues: list[str] = []
    if not api_key:
        issues.append("Add `OPENAI_API_KEY` to your `.env` file.")
    if not MCP_CONFIG_PATH.exists():
        issues.append("`airbnb_mcp.json` is missing from the project root.")
    elif not mcp_command:
        issues.append("No Airbnb MCP command is configured in `airbnb_mcp.json`.")
    elif not mcp_binary_ready:
        issues.append(f"The Airbnb MCP binary was not found at `{mcp_command}`.")

    return {
        "api_key_ready": bool(api_key),
        "mcp_config_ready": MCP_CONFIG_PATH.exists(),
        "mcp_command": mcp_command,
        "mcp_binary_ready": mcp_binary_ready,
        "issues": issues,
    }


def get_chatbot() -> "SimpleChatbot":
    return SimpleChatbot()


class SimpleChatbot:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.client = MCPClient.from_config_file(str(MCP_CONFIG_PATH))
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=self.api_key or None)
        self.agent = MCPAgent(
            llm=self.llm,
            client=self.client,
            max_steps=30,
            system_prompt=CUSTOM_SYSTEM_PROMPT,
        )
        self.initialized = False
        self.last_error = ""

    async def ensure_initialized(self) -> None:
        if not self.api_key:
            raise RuntimeError("OpenAI API key is missing.")
        if not MCP_CONFIG_PATH.exists():
            raise RuntimeError("MCP config file is missing.")
        if not self.initialized:
            await self.agent.initialize()
            self.initialized = True

    async def process_message(
        self,
        message: str,
        chat_history: list[dict[str, Any]] | None = None,
        retry_count: int = 0,
    ) -> str:
        max_retries = 2

        try:
            await self.ensure_initialized()

            formatted_history = []
            for entry in chat_history or []:
                if entry.get("role") == "user":
                    formatted_history.append({"type": "human", "content": entry["content"]})
                elif entry.get("role") == "assistant":
                    formatted_history.append({"type": "ai", "content": entry["content"]})

            result = await self.agent.run(
                message,
                max_steps=30,
                external_history=formatted_history,
            )
            self.last_error = ""
            return str(result)
        except Exception as exc:
            error_message = str(exc)
            self.last_error = error_message
            transient_markers = ("timeout", "connection", "network", "session", "transport")
            if retry_count < max_retries and any(marker in error_message.lower() for marker in transient_markers):
                self.initialized = False
                await asyncio.sleep(1)
                return await self.process_message(message, chat_history, retry_count + 1)
            if retry_count < 1 and "api key" not in error_message.lower() and "config" not in error_message.lower():
                self.initialized = False
                await asyncio.sleep(0.4)
                return await self.process_message(message, chat_history, retry_count + 1)

            if "api key" in error_message.lower():
                return "OpenAI is not configured yet. Add `OPENAI_API_KEY` to `.env` and rerun the app."
            if "timeout" in error_message.lower():
                return "The Airbnb search timed out. Try the same query again in a moment."
            if "connection" in error_message.lower() or "network" in error_message.lower():
                return "A network error interrupted the search. Check connectivity and retry."
            if "mcp" in error_message.lower() or "spawn" in error_message.lower():
                return "The Airbnb MCP server is not responding. Verify the MCP binary path in `airbnb_mcp.json`."
            return (
                "The search failed before listings could be returned. "
                f"Details: {error_message[:220]}"
            )

    async def list_tools(self) -> list[dict[str, Any]]:
        await self.ensure_initialized()
        if hasattr(self.agent._agent, "tools"):
            return [
                {"name": tool.name, "description": tool.description}
                for tool in self.agent._agent.tools
            ]
        return []

    async def close(self):
        if self.client.sessions:
            await self.client.close_all_sessions()


def _init_session_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("pending_query", "")
    st.session_state.setdefault("listing_summary_cache", {})
    st.session_state.setdefault("last_agent_error", "")


def _clear_session_state() -> None:
    st.session_state["chat_history"] = []
    st.session_state["pending_query"] = ""
    st.session_state["listing_summary_cache"] = {}
    st.session_state["last_agent_error"] = ""


def _supports_scraped_summary(url: str) -> bool:
    return url.startswith("http") and "airbnb." in url and "/rooms/" in url


def _summary_label(source: str) -> str:
    if source == "retrieved":
        return "Retrieved Listing Insight"
    return "Metadata-Based Insight"


def _summary_emoji(source: str) -> str:
    if source == "retrieved":
        return "✨"
    return "📝"


def _get_listing_summary(listing: PropertyListing, api_key: str) -> rag.SummaryResult:
    cache: dict[str, rag.SummaryResult] = st.session_state["listing_summary_cache"]
    if listing.cache_key not in cache:
        metadata = {
            "name": listing.name,
            "price": listing.price,
            "rating": listing.rating,
            "desc": listing.desc,
        }
        if _supports_scraped_summary(listing.link):
            cache[listing.cache_key] = rag.summarize_listing(
                listing.link,
                api_key,
                fallback_metadata=metadata,
            )
        else:
            cache[listing.cache_key] = rag.SummaryResult(
                text=rag.build_metadata_summary(metadata),
                source="metadata",
            )
    return cache[listing.cache_key]


def render_header() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <p class="hero-kicker">Project</p>
            <h1 class="hero-title">LLM-MCP Travel Orchestrator</h1>
            <p class="hero-copy">
                Live Airbnb listings in a structured table, followed by one AI summary per property.
            </p>
            <div class="hero-eyebrow-row">
                <span class="hero-chip">🏨 Airbnb Search</span>
                <span class="hero-chip">🧠 RAG Summaries</span>
                <span class="hero-chip">📋 Structured Results</span>
            </div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <p class="hero-stat-label">Search Layer</p>
                    <p class="hero-stat-value">Live Airbnb MCP</p>
                </div>
                <div class="hero-stat">
                    <p class="hero-stat-label">Summary Layer</p>
                    <p class="hero-stat-value">RAG with fallback</p>
                </div>
                <div class="hero-stat">
                    <p class="hero-stat-label">Output Style</p>
                    <p class="hero-stat-value">Structured property findings</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(diagnostics: dict[str, Any]) -> None:
    if diagnostics["issues"]:
        issue_items = " ".join(diagnostics["issues"])
        st.markdown(
            f"""
            <div class="status-banner warning">
                <p class="status-title">Environment Needs Attention</p>
                <p class="status-copy">{escape(issue_items)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="status-banner ready">
            <p class="status-title">System Ready</p>
            <p class="status-copy">
                OpenAI and the Airbnb MCP runtime are available.
                Search responses will render as a structured table, then RAG summaries below each listing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(diagnostics: dict[str, Any], chatbot: SimpleChatbot) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card">
                <p class="sidebar-title">Travel Console</p>
                <p class="sidebar-copy">
                    Reset the session, inspect exposed MCP tools, and verify that the search runtime is healthy.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Clear Conversation", use_container_width=True):
            _clear_session_state()
            st.rerun()

        st.markdown(
            f"""
            <div class="sidebar-card">
                <p class="sidebar-title">Runtime Status</p>
                <p class="sidebar-list">OpenAI key: {"Ready" if diagnostics["api_key_ready"] else "Missing"}</p>
                <p class="sidebar-list">MCP config: {"Ready" if diagnostics["mcp_config_ready"] else "Missing"}</p>
                <p class="sidebar-list">MCP binary: {"Ready" if diagnostics["mcp_binary_ready"] else "Missing"}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if diagnostics["mcp_command"]:
            st.markdown(
                f"""
                <div class="sidebar-card">
                    <p class="sidebar-title">Configured MCP Command</p>
                    <p class="sidebar-copy">{escape(diagnostics["mcp_command"])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("List MCP Tools", use_container_width=True):
            with st.spinner("Loading MCP tools..."):
                try:
                    tools = run_async_task(chatbot.list_tools())
                except Exception as exc:
                    st.error(str(exc))
                else:
                    if not tools:
                        st.info("No tools are currently exposed by the agent.")
                    for tool in tools:
                        st.markdown(
                            f"""
                            <div class="sidebar-card">
                                <p class="sidebar-title">{escape(tool["name"])}</p>
                                <p class="sidebar-copy">{escape(tool["description"][:180])}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


def render_quick_searches() -> None:
    st.markdown(
        """
        <div class="note-card">
            <p class="section-caption">Start Faster</p>
            <h3 class="section-title">Launch Searches</h3>
            <p class="section-copy">
                Start with a preset search or type a more specific budget, city, and date request below.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    first_row = st.columns(3)
    second_row = st.columns(2)
    layout_rows = [first_row, second_row]
    prompt_index = 0
    for row in layout_rows:
        for column in row:
            if prompt_index >= len(QUICK_SEARCHES):
                break
            prompt = QUICK_SEARCHES[prompt_index]
            with column:
                if st.button(prompt, use_container_width=True, key=f"quick_{prompt}"):
                    st.session_state["pending_query"] = prompt
                    st.rerun()
            prompt_index += 1


def render_notes(notes: list[str]) -> None:
    if not notes:
        return
    combined_notes = " ".join(note for note in notes if note)
    st.markdown(
        f"""
        <div class="note-card">
            <p class="section-caption">Context</p>
            <p class="status-title">Search Context</p>
            <p class="section-copy">{escape(combined_notes)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_last_error() -> None:
    if not st.session_state.get("last_agent_error"):
        return
    st.markdown(
        f"""
        <div class="tips-card">
            <p class="section-caption">Diagnostics</p>
            <h3 class="section-title">Latest Search Error</h3>
            <p class="section-copy">{escape(st.session_state["last_agent_error"])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tips(parsed: ParsedAssistantResponse) -> None:
    if not parsed.tips:
        return
    items_html = "".join(f"<li>{escape(item)}</li>" for item in parsed.tips)
    st.markdown(
        f"""
        <div class="tips-card">
            <p class="section-caption">Search Guidance</p>
            <h3 class="section-title">No Direct Matches Yet</h3>
            <p class="section-copy">The search returned guidance rather than property rows.</p>
            <ul class="section-copy">{items_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_property_table(listings: list[PropertyListing]) -> None:
    header_cols = st.columns([3.4, 1.2, 1.0, 3.8, 1.1])
    headers = ["Name", "Price", "Rating", "Description", "Link"]
    for col, header in zip(header_cols, headers):
        with col:
            st.markdown(f'<div class="results-table-header">{header}</div>', unsafe_allow_html=True)

    for listing in listings:
        cols = st.columns([3.4, 1.2, 1.0, 3.8, 1.1])
        rating_text = listing.rating if listing.rating not in {"", "N/A"} else "N/A"

        with cols[0]:
            st.markdown(
                f"""
                <div class="results-row">
                    <div class="results-name">{escape(listing.name)}</div>
                    <div class="results-meta">{escape(listing.desc[:80] + ('...' if len(listing.desc) > 80 else ''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                f'<div class="results-row"><div class="results-cell">{escape(listing.price)}</div></div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                f'<div class="results-row"><div class="results-cell">{escape(rating_text)}</div></div>',
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.markdown(
                f'<div class="results-row"><div class="results-cell">{escape(listing.desc)}</div></div>',
                unsafe_allow_html=True,
            )
        with cols[4]:
            link_markup = (
                f'<a href="{escape(listing.link)}" target="_blank">View Listing</a>'
                if listing.link
                else 'Unavailable'
            )
            st.markdown(
                f'<div class="results-row results-link">{link_markup}</div>',
                unsafe_allow_html=True,
            )


def _render_summary_card(
    index: int,
    listing: PropertyListing,
    summary_result: rag.SummaryResult | None,
) -> None:
    source = summary_result.source if summary_result else "unavailable"
    source_class = "retrieved" if source == "retrieved" else "metadata"
    source_badge_class = "source" if source == "retrieved" else "metadata"
    source_label = _summary_label(source) if summary_result else "Summary unavailable"
    rating_text = listing.rating if listing.rating not in {"", "N/A"} else "N/A"
    summary_text = summary_result.text if summary_result else "Summary unavailable."
    subtitle = escape(listing.desc if listing.desc else "No extra property description provided.")
    source_hint = "RAG summary" if source == "retrieved" else "Metadata summary"
    link_html = (
        f'<a class="summary-link" href="{escape(listing.link)}" target="_blank">🔗 Open Airbnb listing</a>'
        if listing.link
        else ""
    )

    st.markdown(
        f"""
        <div class="summary-card {source_class}">
            <div class="summary-header">
                <span class="summary-number">{index}</span>
                <div>
                    <p class="summary-title">{_summary_emoji(source)} {escape(listing.name)}</p>
                    <p class="summary-subtitle">{subtitle}</p>
                </div>
            </div>
            <div class="summary-badges">
                <span class="summary-badge">💵 {escape(listing.price or 'N/A')}</span>
                <span class="summary-badge">⭐ {escape(rating_text)}</span>
                <span class="summary-badge {source_badge_class}">🧠 {escape(source_label)}</span>
            </div>
            <p class="summary-text">{escape(summary_text)}</p>
            <div class="summary-footer">
                {link_html}
                <span class="summary-footer-note">📌 {escape(source_hint)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_property_listings(parsed: ParsedAssistantResponse, api_key: str) -> None:
    render_notes(parsed.notes)
    st.markdown(
        f"""
        <div class="note-card">
            <p class="section-caption">Results</p>
            <h3 class="section-title">Property Findings</h3>
            <p class="section-copy">
                {len(parsed.listings)} listing{"s" if len(parsed.listings) != 1 else ""} returned.
                The table is the primary result surface. AI summaries are generated below for every returned listing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_property_table(parsed.listings)

    st.markdown(
        """
        <div class="note-card">
            <p class="section-caption">Insights</p>
            <h3 class="section-title">AI Summary Layer</h3>
            <p class="section-copy">
                Each listing below includes a RAG summary when the property page can be retrieved, with metadata fallback otherwise.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for index, listing in enumerate(parsed.listings, start=1):
        summary_result = _get_listing_summary(listing, api_key)
        _render_summary_card(index, listing, summary_result)


def render_response(response: str, api_key: str) -> bool:
    parsed = parse_assistant_response(response)
    if parsed.listings:
        render_property_listings(parsed, api_key)
        return True
    if parsed.tips:
        render_notes(parsed.notes)
        render_tips(parsed)
        return True
    return False


def render_chat_history(api_key: str) -> None:
    for message in st.session_state["chat_history"]:
        with st.chat_message("assistant" if message["role"] != "user" else "user"):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message["role"] == "assistant":
                if not render_response(message["content"], api_key):
                    st.markdown(message["content"])
            else:
                st.error(message["content"])


def main() -> None:
    warnings.filterwarnings("ignore", category=ResourceWarning)
    load_dotenv()
    st.markdown(APP_CSS, unsafe_allow_html=True)
    _init_session_state()

    diagnostics = get_runtime_diagnostics()
    chatbot = get_chatbot()

    render_sidebar(diagnostics, chatbot)
    render_header()
    render_status_banner(diagnostics)
    render_last_error()

    if not st.session_state["chat_history"]:
        render_quick_searches()

    render_chat_history(chatbot.api_key)

    user_input = st.chat_input("Ask for listings by city, budget, dates, or traveler count...")
    if not user_input and st.session_state["pending_query"]:
        user_input = st.session_state["pending_query"]
        st.session_state["pending_query"] = ""

    if user_input and user_input.strip():
        st.session_state["chat_history"].append(
            {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M"),
            }
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching live Airbnb data and compiling findings..."):
                try:
                    response = run_async_task(
                        chatbot.process_message(user_input, st.session_state["chat_history"][:-1])
                    )
                    st.session_state["last_agent_error"] = chatbot.last_error
                    if not render_response(response, chatbot.api_key):
                        st.markdown(response)
                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().strftime("%H:%M"),
                        }
                    )
                except Exception:
                    error_text = "The request failed before a response could be rendered."
                    st.session_state["last_agent_error"] = error_text
                    st.error(error_text)
                    st.session_state["chat_history"].append(
                        {
                            "role": "error",
                            "content": error_text,
                            "timestamp": datetime.now().strftime("%H:%M"),
                        }
                    )


if __name__ == "__main__":
    main()
