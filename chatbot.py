"""
LLM-MCP Travel Orchestrator — Streamlit chatbot using MCP tools and OpenAI GPT-4o-mini.
"""

import os
import re
import warnings
import asyncio
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
from mcp_use import MCPAgent, MCPClient

st.set_page_config(
    page_title="LLM-MCP Travel Orchestrator",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _get_default_dates() -> tuple[str, str]:
    """Return checkin (tomorrow) and checkout (day after) — a 1-night stay.
    This ensures maxPrice == per-night price, since Airbnb's price_max is the TOTAL stay price."""
    from datetime import timedelta
    checkin  = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
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
Example: User says "under $100/night" → set maxPrice=100 (1-night stay = $100 total = $100/night).
If the user specifies a different number of nights N, multiply: maxPrice = budget_per_night × N.

== TOOL PARAMETERS (airbnb_search) ==
Supported: location, checkin, checkout, adults, children, minPrice, maxPrice
NOT supported: pool, amenities, gym, wifi, or any other feature/amenity filters

== RULES ==
1. ALWAYS include checkin="{_CHECKIN}" and checkout="{_CHECKOUT}" unless the user specifies dates.
2. ALWAYS apply minPrice/maxPrice when the user mentions a budget or price limit.
3. If the user asks for amenities (pool, gym, wifi, etc.) — search by location only,
   then note in your reply that amenity filtering is not supported by the search tool.
4. If a price-filtered search returns 0 results, DO NOT silently remove the filter.
   Instead, tell the user: "No listings found within $X/night in [city]. The market may be
   above budget — here are nearby options at higher prices:" then retry without the price filter.
5. Return results as a pipe-delimited table with EXACTLY 5 columns per row (no header, no separator):
   Name | Price | Rating | Description | URL
6. URL must be the raw Airbnb URL (https://www.airbnb.com/rooms/...) — NO markdown link syntax.
7. Price must be formatted as per-night (e.g. $97/night) even if the tool returns a total price.
   To convert: divide the total returned price by number of nights.
8. Only say "No listings found" if retrying without any filters also returns 0 results.
"""


# ── Chatbot ────────────────────────────────────────────────────────────────────

class SimpleChatbot:
    def __init__(self):
        load_dotenv()
        self.client = MCPClient.from_config_file(
            os.path.join(os.path.dirname(__file__), "airbnb_mcp.json")
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        self.agent = MCPAgent(
            llm=self.llm,
            client=self.client,
            max_steps=30,
            system_prompt=CUSTOM_SYSTEM_PROMPT
        )
        self.initialized = False

    async def process_message(self, message: str, chat_history=None) -> str:
        import traceback
        try:
            if not self.initialized:
                await self.agent.initialize()
                self.initialized = True
            formatted_history = []
            if chat_history:
                for msg in chat_history:
                    if msg["role"] == "user":
                        formatted_history.append({"type": "human", "content": msg["content"]})
                    elif msg["role"] == "assistant":
                        formatted_history.append({"type": "ai", "content": msg["content"]})
            result = await self.agent.run(message, max_steps=30, external_history=formatted_history)
            return str(result)
        except Exception as e:
            print(f"Error: {e}\n{traceback.format_exc()}")
            return "I encountered an error while processing your request. Please try again."

    async def list_tools(self) -> List[Dict[str, Any]]:
        if not self.initialized:
            await self.agent.initialize()
            self.initialized = True
        if hasattr(self.agent._agent, "tools"):
            return [{"name": t.name, "description": t.description} for t in self.agent._agent.tools]
        return []

    async def close(self):
        if self.client.sessions:
            await self.client.close_all_sessions()


# ── Parsing ────────────────────────────────────────────────────────────────────

def _extract_url(text: str):
    """Extract URL from raw URL or markdown link [text](url), return (clean_text, url)."""
    # Try markdown link format: [text](url)
    md = re.search(r'\[([^\]]*)\]\((https?://[^\)]+)\)', text)
    if md:
        return md.group(1).strip() or text, md.group(2).strip()
    # Try bare URL
    bare = re.search(r'(https?://\S+)', text)
    if bare:
        url = bare.group(1).rstrip(')')
        clean = text.replace(url, '').strip().strip('|').strip()
        return clean or text, url
    return text, ""

def _is_separator(line: str) -> bool:
    return bool(re.match(r"^\|?[\s\-:]+(\|[\s\-:]+)+\|?$", line))

def _is_header(line: str) -> bool:
    lower = line.lower()
    keywords = ["name", "price", "rating", "description", "link", "url", "property"]
    matches = sum(1 for kw in keywords if kw in lower)
    return matches >= 2  # at least 2 column header keywords

def _parse_row(line: str) -> dict | None:
    """Parse a pipe-delimited row into a property dict."""
    # Split on pipes, strip empties from leading/trailing |
    parts = [p.strip() for p in line.split("|")]
    parts = [p for p in parts if p]

    if len(parts) < 2:
        return None

    # Extract any URL from any cell
    url = ""
    cleaned = []
    for p in parts:
        text, found_url = _extract_url(p)
        if found_url and not url:
            url = found_url
            cleaned.append(text)
        else:
            cleaned.append(p)
    parts = [p for p in cleaned if p.strip()]

    # Remove the URL column if it was a standalone URL cell
    parts = [p for p in parts if not re.match(r'^https?://', p.strip())]

    while len(parts) < 4:
        parts.append("")

    name   = parts[0].strip()
    price  = parts[1].strip() if len(parts) > 1 else ""
    rating = parts[2].strip() if len(parts) > 2 else ""
    desc   = " ".join(parts[3:]).strip() if len(parts) > 3 else ""

    # Clean up any leftover markdown from desc
    desc, extra_url = _extract_url(desc)
    if extra_url and not url:
        url = extra_url

    if not name or not any([price, rating, url]):
        return None

    # Clean numeric rating
    rating_clean = re.sub(r"[^\d.]", "", rating)
    try:
        rating_num = float(rating_clean)
    except ValueError:
        rating_num = 0.0

    return {
        "name":       name,
        "price":      price  or "N/A",
        "rating":     rating_clean or rating or "N/A",
        "rating_num": rating_num,
        "desc":       desc   or "—",
        "link":       url,
    }


# ── Rendering ──────────────────────────────────────────────────────────────────

def _rating_color(val: float) -> str:
    if val >= 4.7: return "#22c55e"   # green
    if val >= 4.3: return "#f59e0b"   # amber
    return "#ef4444"                  # red

def render_response(response: str) -> bool:
    """Render listings as cards or tips. Returns True if something was rendered."""
    lines = [l.strip() for l in response.splitlines() if l.strip()]

    # ── Collect listings ───────────────────────────────────────────────────
    listings = []
    for line in lines:
        if "|" not in line:
            continue
        if _is_separator(line) or _is_header(line):
            continue
        row = _parse_row(line)
        if row:
            listings.append(row)

    # ── Collect tips ───────────────────────────────────────────────────────
    tips = []
    for line in lines:
        if re.match(r"^\d+[\.\)]\s", line):
            tips.append(re.sub(r"^\d+[\.\)]\s*", "", line).strip())
        elif re.match(r"^[-*•]\s", line):
            tips.append(re.sub(r"^[-*•]\s*", "", line).strip())

    # ── Render property table ──────────────────────────────────────────────
    if listings:
        # Build all rows first so the entire table is one st.markdown call
        rows_html = ""
        for i, p in enumerate(listings):
            rc = _rating_color(p["rating_num"])

            # Rating cell
            if p["rating"] not in ("N/A", "—") and p["rating_num"] > 0:
                rating_cell = (
                    f'<span style="display:inline-flex;align-items:center;gap:5px;">'
                    f'<span style="background:{rc};color:#fff;padding:2px 9px;'
                    f'border-radius:20px;font-weight:700;font-size:0.78rem;">★ {p["rating"]}</span>'
                    f'</span>'
                )
            else:
                rating_cell = '<span style="color:#334155;font-size:0.78rem;">—</span>'

            # Name cell — linked if URL available
            name_cell = (
                f'<a href="{p["link"]}" target="_blank" class="prop-name" '
                f'style="color:#cbd5e1;font-weight:600;font-size:0.88rem;'
                f'text-decoration:none;display:block;line-height:1.35;transition:color 0.15s;">'
                f'{p["name"]}</a>'
                if p["link"] else
                f'<span style="color:#cbd5e1;font-weight:600;font-size:0.88rem;">{p["name"]}</span>'
            )

            # Description — only show if not empty or meaningless
            desc_html = (
                f'<span style="color:#475569;font-size:0.76rem;display:block;'
                f'margin-top:3px;line-height:1.3;">{p["desc"]}</span>'
                if p["desc"] not in ("—", "", "N/A") else ""
            )

            # Price cell
            price_cell = (
                f'<span style="color:#4ade80;font-weight:700;font-size:0.95rem;">{p["price"]}</span>'
                if re.search(r'\d', p["price"]) else
                f'<span style="color:#334155;font-size:0.82rem;">—</span>'
            )

            # Book button
            book_btn = (
                f'<a href="{p["link"]}" target="_blank" class="view-btn" '
                f'style="display:inline-block;background:#FF5A5F;color:#fff;'
                f'padding:6px 16px;border-radius:6px;text-decoration:none;'
                f'font-weight:600;font-size:0.76rem;white-space:nowrap;'
                f'letter-spacing:0.02em;transition:background 0.15s;">View →</a>'
                if p["link"] else
                '<span style="color:#252a3a;font-size:0.76rem;">—</span>'
            )

            row_bg = "#111420" if i % 2 == 0 else "#161922"
            rows_html += f"""
            <tr style="background:{row_bg};">
                <td style="padding:13px 10px 13px 16px;color:#475569;font-size:0.75rem;
                           font-weight:600;width:32px;text-align:center;
                           border-bottom:1px solid #1a1d2e;">{i+1}</td>
                <td style="padding:13px 16px;border-bottom:1px solid #1a1d2e;">
                    {name_cell}{desc_html}
                </td>
                <td style="padding:13px 16px;border-bottom:1px solid #1a1d2e;
                           white-space:nowrap;">{price_cell}</td>
                <td style="padding:13px 16px;border-bottom:1px solid #1a1d2e;
                           white-space:nowrap;">{rating_cell}</td>
                <td style="padding:13px 16px;border-bottom:1px solid #1a1d2e;
                           text-align:right;">{book_btn}</td>
            </tr>"""

        table_html = f"""
        <p style="color:#475569;font-size:0.77rem;margin:0 0 8px 0;letter-spacing:0.01em;">
            🔍 {len(listings)} properties · Airbnb MCP
        </p>
        <div style="overflow-x:auto;border-radius:10px;border:1px solid #1a1d2e;
                    box-shadow:0 4px 20px rgba(0,0,0,0.4);">
            <table class="prop-table"
                   style="width:100%;border-collapse:collapse;font-family:system-ui,sans-serif;">
                <thead>
                    <tr style="background:#0a0d14;border-bottom:1px solid #1a1d2e;">
                        <th style="padding:11px 10px 11px 16px;text-align:center;color:#334155;
                                   font-size:0.68rem;font-weight:600;letter-spacing:0.07em;
                                   text-transform:uppercase;width:32px;">#</th>
                        <th style="padding:11px 16px;text-align:left;color:#475569;
                                   font-size:0.68rem;font-weight:600;letter-spacing:0.07em;
                                   text-transform:uppercase;">Property</th>
                        <th style="padding:11px 16px;text-align:left;color:#475569;
                                   font-size:0.68rem;font-weight:600;letter-spacing:0.07em;
                                   text-transform:uppercase;white-space:nowrap;">Price / night</th>
                        <th style="padding:11px 16px;text-align:left;color:#475569;
                                   font-size:0.68rem;font-weight:600;letter-spacing:0.07em;
                                   text-transform:uppercase;">Rating</th>
                        <th style="padding:11px 16px;text-align:right;color:#475569;
                                   font-size:0.68rem;font-weight:600;letter-spacing:0.07em;
                                   text-transform:uppercase;">Link</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>"""

        st.markdown(table_html, unsafe_allow_html=True)
        return True

    # ── Render tips ────────────────────────────────────────────────────────
    if tips:
        items_html = "".join(
            f'<li style="color:#94a3b8;margin-bottom:7px;line-height:1.5;">{t}</li>'
            for t in tips
        )
        st.markdown(
            f"""<div style="background:#1a1d27;border:1px solid #252a3a;
                            border-left:3px solid #f59e0b;border-radius:10px;
                            padding:16px 20px;margin:4px 0;">
                <p style="color:#f59e0b;font-weight:600;margin:0 0 10px 0;font-size:0.88rem;">
                    💡 No listings found — a few tips:
                </p>
                <ul style="margin:0;padding-left:16px;">{items_html}</ul>
            </div>""",
            unsafe_allow_html=True
        )
        return True

    return False


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <style>
        /* Base */
        [data-testid="stAppViewContainer"] > .main { background:#0f1117; }
        section[data-testid="stSidebar"] { background:#13161f; border-right:1px solid #1a1d27; }

        /* Chat input */
        [data-testid="stChatInput"] textarea {
            background:#1a1d27 !important;
            color:#e2e8f0 !important;
            border:1px solid #252a3a !important;
            border-radius:10px !important;
            font-size:0.93rem !important;
        }
        [data-testid="stChatInput"] textarea::placeholder { color:#334155 !important; }

        /* Chat messages */
        [data-testid="stChatMessage"] {
            background:transparent !important;
            border:none !important;
            padding:4px 0 !important;
        }

        /* Sidebar buttons */
        .stButton > button {
            border-radius:7px;
            font-weight:500;
            font-size:0.82rem;
            border:1px solid #1e2235;
            background:#151820;
            color:#64748b;
            width:100%;
            text-align:left;
            padding:7px 12px;
            transition:all 0.15s ease;
        }
        .stButton > button:hover {
            background:#1e2235;
            border-color:#334155;
            color:#94a3b8;
        }

        /* Markdown text */
        [data-testid="stMarkdownContainer"] p { color:#94a3b8; }
        [data-testid="stMarkdownContainer"] li { color:#94a3b8; }

        /* Property table: row hover */
        .prop-table tbody tr { transition:background 0.12s ease; }
        .prop-table tbody tr:hover td { background:#1e2235 !important; }

        /* Property table: name link hover */
        .prop-table a.prop-name:hover {
            color:#93c5fd !important;
            text-decoration:underline !important;
        }

        /* Property table: view button hover */
        .prop-table a.view-btn:hover { background:#e0474c !important; }

        /* Scrollbar */
        ::-webkit-scrollbar { width:5px; height:5px; }
        ::-webkit-scrollbar-track { background:#0f1117; }
        ::-webkit-scrollbar-thumb { background:#1e2235; border-radius:4px; }
        ::-webkit-scrollbar-thumb:hover { background:#2d3347; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:18px 0 14px 0;border-bottom:1px solid #1a1d27;margin-bottom:18px;">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:1.9rem;">🏨</span>
            <div>
                <h1 style="margin:0;font-size:1.5rem;font-weight:700;color:#e2e8f0;">
                    LLM-MCP Travel Orchestrator
                </h1>
                <p style="margin:2px 0 0 0;color:#475569;font-size:0.82rem;">
                    Real-time Airbnb search · GPT-4o-mini · Model Context Protocol
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = SimpleChatbot()
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = ""
    chatbot = st.session_state["chatbot"]

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<p style='color:#e2e8f0;font-weight:600;font-size:0.95rem;margin:0 0 12px 0;'>"
            "🏨 Travel Assistant</p>",
            unsafe_allow_html=True
        )

        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

        st.markdown(
            "<p style='color:#475569;font-size:0.74rem;font-weight:600;"
            "margin:16px 0 6px 0;'>Quick searches</p>",
            unsafe_allow_html=True
        )
        quick_queries = [
            "5 affordable Airbnbs in Paris",
            "3 Airbnbs in Bali under $120",
            "Apartments near Central Park NYC",
            "Budget stays in Tokyo under $80",
            "Beachfront rentals in Barcelona",
        ]
        for q in quick_queries:
            if st.button(q, use_container_width=True, key=f"q_{q}"):
                st.session_state["pending_query"] = q
                st.rerun()

        st.markdown(
            "<p style='color:#475569;font-size:0.74rem;font-weight:600;"
            "margin:16px 0 6px 0;'>Tools</p>",
            unsafe_allow_html=True
        )
        if st.button("🔧  List MCP Tools", use_container_width=True):
            chatbot_obj = st.session_state.get("chatbot")
            if chatbot_obj:
                with st.spinner("Fetching..."):
                    try:
                        tools = asyncio.run(chatbot_obj.list_tools())
                        for t in tools:
                            st.markdown(
                                f"<p style='color:#4ea1f7;font-size:0.8rem;font-weight:600;"
                                f"margin:4px 0 1px 0;'>⚙ {t['name']}</p>"
                                f"<p style='color:#475569;font-size:0.74rem;margin:0 0 6px 12px;'>"
                                f"{t['description'][:72]}…</p>",
                                unsafe_allow_html=True
                            )
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown(
            "<p style='color:#334155;font-size:0.7rem;margin-top:20px;text-align:center;"
            "border-top:1px solid #1a1d27;padding-top:10px;'>OpenAI · LangChain · MCP</p>",
            unsafe_allow_html=True
        )

    # ── Welcome ──────────────────────────────────────────────────────────────
    if not st.session_state["chat_history"]:
        with st.chat_message("assistant"):
            st.markdown(
                "👋 **Welcome!** I search live Airbnb listings in any city using AI "
                "and the Model Context Protocol.\n\n"
                "Try a quick search from the sidebar, or type your own below."
            )

    # ── Chat history ─────────────────────────────────────────────────────────
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                rendered = render_response(msg["content"])
                if not rendered:
                    st.markdown(msg["content"])
        elif msg["role"] == "error":
            with st.chat_message("assistant"):
                st.error(msg["content"])

    # ── Input (typed or from sidebar chip) ───────────────────────────────────
    user_input = st.chat_input("Search accommodations in any city...")

    # Pick up a pending query from a sidebar button click
    if not user_input and st.session_state["pending_query"]:
        user_input = st.session_state["pending_query"]
        st.session_state["pending_query"] = ""

    if user_input and user_input.strip():
        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M"),
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching Airbnb listings..."):
                try:
                    response = asyncio.run(
                        chatbot.process_message(user_input, st.session_state["chat_history"])
                    )
                    rendered = render_response(response)
                    if not rendered:
                        st.markdown(response)
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M"),
                    })
                except Exception as e:
                    err = "I encountered an error. Please try rephrasing your question."
                    st.error(err)
                    st.session_state["chat_history"].append({
                        "role": "error",
                        "content": err,
                        "timestamp": datetime.now().strftime("%H:%M"),
                    })


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ResourceWarning)
    main()
