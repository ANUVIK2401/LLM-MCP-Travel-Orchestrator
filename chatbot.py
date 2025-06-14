"""
Enhanced Streamlit chatbot example using MCP tools and OpenAI GPT-4o mini.

This example demonstrates a beautiful, user-friendly chatbot with a Streamlit interface
that can handle conversations and perform various tasks using MCP tools and OpenAI GPT-4o mini.
"""

import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from mcp_use import MCPAgent, MCPClient

# Set page config
st.set_page_config(
    page_title="LLM-MCP Travel Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom system prompt to encourage broad city support
CUSTOM_SYSTEM_PROMPT = """
You are an assistant with access to these tools:

{{tool_descriptions}}

When users ask about accommodations, properties, or hotels in any city, ALWAYS use the Airbnb tool to search for properties. Return results as a table with columns: Name, Price, Rating, Description, Link. If no properties are found, say "No listings found, but here are some tips:" and then provide tips. Do not provide tips unless no listings are available. Do not summarize or reformat the table; return it as a markdown table or pipe-delimited text.
"""

USER_AVATAR = "🧑"
BOT_AVATAR = "🤖"

class SimpleChatbot:
    """A simple chatbot that can handle conversations and perform tasks."""
    def __init__(self):
        load_dotenv()
        self.client = MCPClient.from_config_file(os.path.join(os.path.dirname(__file__), "airbnb_mcp.json"))
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
            # Prepare chat history for context
            formatted_history = []
            if chat_history:
                for msg in chat_history:
                    if msg['role'] == 'user':
                        formatted_history.append({"type": "human", "content": msg['content']})
                    elif msg['role'] == 'assistant':
                        formatted_history.append({"type": "ai", "content": msg['content']})
            result = await self.agent.run(message, max_steps=30, external_history=formatted_history)
            return str(result)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error: {e}\nTraceback:\n{tb}")
            return (
                "I apologize, but I encountered an error while processing your request. "
                "Please try rephrasing your question or ask about another city."
            )

    async def list_tools(self) -> List[Dict[str, Any]]:
        if not self.initialized:
            await self.agent.initialize()
            self.initialized = True
        if hasattr(self.agent._agent, "tools"):
            return [
                {"name": tool.name, "description": tool.description}
                for tool in self.agent._agent.tools
            ]
        return []

    async def close(self):
        if self.client.sessions:
            await self.client.close_all_sessions()

def render_property_listings(response: str):
    """Render property listings as a visually consistent, professional table or beautiful tips."""
    import re
    import streamlit as st

    # Helper: Detect if a line is a tip/advice
    def is_tip(line):
        tip_starters = [
            "Consider", "Look for", "Use", "Off Peak", "Last Minute", "Check Alternative", "Tip:", "Advice:", "Try:", "Pro tip:", "Note:", "*", "-"
        ]
        return any(line.strip().startswith(starter) for starter in tip_starters)

    # Helper: Parse property into dict with consistent keys
    def parse_property(line):
        parts = [p.strip() for p in re.split(r"\s*\|\s*", line)]
        link = ''
        for i, p in enumerate(parts):
            if p.startswith("http"): link = p; parts[i] = ''
        while len(parts) < 5:
            parts.append('')
        if len(parts) > 5:
            parts = parts[:4] + [link or parts[4]]
        name, price, rating, type_desc, link_col = parts
        link = link_col or link
        if not any([name, price, rating, type_desc, link]):
            return None
        if name and not any([price, rating, type_desc, link]):
            return None
        return {
            "name": name or '—',
            "price": price or '—',
            "rating": rating or '—',
            "type_desc": type_desc or '—',
            "link": link or ''
        }

    lines = [l.strip() for l in response.splitlines() if l.strip()]
    tips = [l for l in lines if is_tip(l) or re.match(r"^\*\*.+\*\*", l)]
    listings = []
    for line in lines:
        if ("|" in line and not line.startswith("|")) or re.match(r"\d+\.\s", line):
            prop = parse_property(line)
            if prop:
                listings.append(prop)

    # --- Enhanced Tips/Advice UI ---
    if not listings and tips:
        st.markdown(
            '''<div style="background: #e7f3fe; color: #31708f; border-radius: 18px; padding: 24px 28px; margin-bottom: 1.5rem; box-shadow: 0 4px 24px #0001;">
                <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 1.2rem;">
                    <span style="font-size: 2.2rem;">💡</span>
                    <span style="font-size: 1.35rem; font-weight: 700; letter-spacing: 0.01em;">No listings found, but here are some helpful tips!</span>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 16px;">
            ''', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(
                f'''<div style="background: #fafdff; color: #31708f; border-radius: 12px; padding: 16px 18px; min-width: 220px; margin-bottom: 0.5rem; box-shadow: 0 2px 8px #0001; display: flex; align-items: flex-start; gap: 10px;">
                        <span style="font-size: 1.3rem; margin-right: 8px;">📝</span>
                        <span style="font-size: 1.08rem;">{tip}</span>
                    </div>''', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        return True

    # --- Enhanced Property Table UI ---
    if listings:
        st.markdown("""
        <style>
        .property-table-wrap { overflow-x: auto; margin-bottom: 2rem; }
        .property-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 12px;
            font-family: 'system-ui', Arial, sans-serif;
            font-size: 1.09rem;
            box-shadow: 0 6px 32px #0002;
            background: #191d26;
            border-radius: 22px;
        }
        .property-table th, .property-table td {
            padding: 18px 22px;
            text-align: left;
            font-weight: 400;
        }
        .property-table th {
            background: #23272f;
            color: #fff;
            font-size: 1.18rem;
            border-radius: 14px 14px 0 0;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .property-table td {
            background: #23272f;
            color: #f5f5f5;
            border-radius: 12px;
            vertical-align: middle;
            border-bottom: 1px solid #232b36;
            max-width: 320px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .property-table tr:nth-child(even) td {
            background: #232b36;
        }
        .property-table tr:hover td {
            background: #2a3140;
            transition: background 0.2s;
        }
        .property-table a {
            color: #4ea1f7;
            text-decoration: underline;
            font-weight: 500;
            transition: color 0.2s;
        }
        .property-table a:hover {
            color: #1e70c1;
        }
        .property-table .star {
            color: #f7c948;
            font-size: 1.13em;
            margin-right: 2px;
        }
        .property-table .rating-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1.01em;
            background: #2e7d32;
            color: #fff;
            margin-left: 4px;
        }
        .property-table .rating-badge.low {
            background: #b71c1c;
        }
        .property-table .view-btn {
            background: #31708f;
            color: #fff;
            padding: 9px 18px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.05rem;
            margin-left: 0;
            display: inline-block;
            transition: background 0.2s;
        }
        .property-table .view-btn:hover {
            background: #1e70c1;
        }
        .property-table .ellipsis {
            max-width: 220px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            display: inline-block;
            vertical-align: bottom;
        }
        </style>
        <div class="property-table-wrap">
        <table class="property-table">
            <tr>
                <th>🏠 Property Name</th>
                <th>💲 Price (per night)</th>
                <th>★ Rating</th>
                <th>🏷️ Type/Description</th>
                <th>🔗 Link</th>
            </tr>
        """, unsafe_allow_html=True)
        for prop in listings:
            # Tooltip for long names
            name_html = (
                f'<span class="ellipsis" title="{prop["name"]}"><a href="{prop["link"]}" target="_blank">{prop["name"]}</a></span>' if prop["link"] else f'<span class="ellipsis" title="{prop["name"]}">{prop["name"]}</span>'
            )
            # Rating badge
            rating_val = 0
            try:
                rating_val = float(prop["rating"])
            except Exception:
                pass
            badge_class = "rating-badge" + (" low" if rating_val and rating_val < 4.5 else "")
            rating_html = (
                f'<span class="star">★</span> <span class="{badge_class}">{prop["rating"]}</span>' if prop["rating"] not in ['—', ''] else '—'
            )
            view_btn = (
                f'<a href="{prop["link"]}" target="_blank" class="view-btn" title="Open Airbnb listing in new tab">View Listing</a>' if prop["link"] else '—'
            )
            st.markdown(
                f"""
                <tr>
                    <td>{name_html}</td>
                    <td>{prop['price']}</td>
                    <td>{rating_html}</td>
                    <td>{prop['type_desc']}</td>
                    <td>{view_btn}</td>
                </tr>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</table></div>", unsafe_allow_html=True)
        return True
    return False

def main():
    st.markdown(
        """
        <style>
        .stChatMessage { font-size: 1.1em; }
        .user-msg { background: #23272f; color: #fff; border-radius: 16px; padding: 12px 18px; margin-bottom: 8px; display: flex; align-items: flex-start; gap: 10px; }
        .bot-msg { background: #f5f5f5; color: #222; border-radius: 16px; padding: 12px 18px; margin-bottom: 8px; display: flex; align-items: flex-start; gap: 10px; }
        .error-msg { background: #ffdddd; color: #a00; border-radius: 16px; padding: 12px 18px; margin-bottom: 8px; }
        .info-msg { background: #e7f3fe; color: #31708f; border-radius: 16px; padding: 12px 18px; margin-bottom: 8px; }
        .stButton>button { border-radius: 8px; font-weight: 600; }
        .stTextInput>div>div>input { border-radius: 8px; }
        .stSidebar { background: #f7f7fa; }
        .avatar { font-size: 1.7rem; margin-right: 8px; }
        .timestamp { font-size: 0.85rem; color: #888; margin-left: 8px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 18px; margin-bottom: 0.5rem;">
            <span style="font-size: 2.7rem;">🏨</span>
            <div>
                <h1 style="margin-bottom: 0;">LLM-MCP Travel Orchestrator</h1>
                <div style="font-size: 1.2rem; color: #666;">Find the perfect accommodation in any city! <span style='font-size:1.2rem;'>🌍</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
            <b>I can help you find:</b>
            <ul>
                <li>Available properties in any city or neighborhood</li>
                <li>Properties within your budget</li>
                <li>Properties with specific amenities</li>
                <li>And more!</li>
            </ul>
            <span style="color: #31708f; font-size: 1rem;">Ask about any city worldwide for recommendations and details.</span>
        </div>
        <div style="margin-top: 1rem; margin-bottom: 1.5rem; background: #e7f3fe; color: #31708f; border-radius: 12px; padding: 12px 18px;">
            <b>💡 Tip:</b> Try asking for "affordable places in Paris", "properties with a pool in Tokyo", or "apartments near Central Park in New York".
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Try These Prompts Section ---
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <b>Try these prompts:</b>
            <ul style="margin-top: 0.5em;">
                <li><a href="#" onclick="window.parent.document.querySelector('input[type=text]').value='Show me 5 affordable Airbnbs in Paris with price, rating, and link.'; window.parent.document.querySelector('button[type=submit]').click(); return false;">Show me 5 affordable Airbnbs in Paris with price, rating, and link.</a></li>
                <li><a href="#" onclick="window.parent.document.querySelector('input[type=text]').value='List 3 properties in San Francisco with a pool, including price and link.'; window.parent.document.querySelector('button[type=submit]').click(); return false;">List 3 properties in San Francisco with a pool, including price and link.</a></li>
                <li><a href="#" onclick="window.parent.document.querySelector('input[type=text]').value='Find me Airbnbs in Tokyo with a gym, and show the results in a table.'; window.parent.document.querySelector('button[type=submit]').click(); return false;">Find me Airbnbs in Tokyo with a gym, and show the results in a table.</a></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("Options")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state['chat_history'] = []
            st.experimental_rerun()
        st.markdown("---")
        st.markdown("### Example Questions")
        st.markdown(
            """
            - What are some affordable places to stay in Paris?
            - Show me properties in the Mission District, San Francisco
            - Find me a place with a pool and gym in Tokyo
            - What's available near Central Park in New York?
            """
        )
        st.markdown("---")
        if st.button("List Available Tools", use_container_width=True):
            chatbot = st.session_state.get('chatbot')
            if chatbot:
                with st.spinner("Fetching tools..."):
                    try:
                        tool_list = asyncio.run(chatbot.list_tools())
                        if tool_list:
                            tool_md = "\n".join(
                                f"- **{tool['name']}**: {tool['description']}" for tool in tool_list
                            )
                            st.info(f"**Available tools:**\n{tool_md}")
                        else:
                            st.warning("No tools available.")
                    except Exception as e:
                        st.error(f"Error fetching tools: {e}")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'chatbot' not in st.session_state:
        st.session_state['chatbot'] = SimpleChatbot()
    chatbot = st.session_state['chatbot']

    # Welcome message
    if len(st.session_state['chat_history']) == 0:
        st.markdown(
            f'<div class="bot-msg"><span class="avatar">{BOT_AVATAR}</span> Welcome! 👋 I can help you find accommodations in any city. Just type your question below to get started.<span class="timestamp">{datetime.now().strftime("%H:%M")}</span></div>',
            unsafe_allow_html=True
        )

    # Chat interface
    user_input = st.chat_input("Ask about accommodations in any city...")

    # Display chat history
    for msg in st.session_state['chat_history']:
        timestamp = msg.get('timestamp', datetime.now().strftime("%H:%M"))
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(f'<div class="user-msg"><span class="avatar">{USER_AVATAR}</span> {msg["content"]}<span class="timestamp">{timestamp}</span></div>', unsafe_allow_html=True)
        elif msg['role'] == 'assistant':
            with st.chat_message("assistant"):
                st.markdown(f'<div class="bot-msg"><span class="avatar">{BOT_AVATAR}</span> ', unsafe_allow_html=True)
                # Try to render property listings as a table/cards if possible
                rendered = render_property_listings(msg["content"])
                if not rendered:
                    st.warning("Sorry, I couldn't find any listings or tips for your query.")
                    st.markdown(f'{msg["content"]}<span class="timestamp">{timestamp}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        elif msg['role'] == 'error':
            st.markdown(f'<div class="error-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        elif msg['role'] == 'info':
            st.markdown(f'<div class="info-msg">{msg["content"]}</div>', unsafe_allow_html=True)

    # Handle new user input
    if user_input and user_input.strip():
        st.session_state['chat_history'].append({'role': 'user', 'content': user_input, 'timestamp': datetime.now().strftime("%H:%M")})
        with st.chat_message("user"):
            st.markdown(f'<div class="user-msg"><span class="avatar">{USER_AVATAR}</span> {user_input}<span class="timestamp">{datetime.now().strftime("%H:%M")}</span></div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            with st.spinner("Searching for accommodations..."):
                try:
                    response = asyncio.run(chatbot.process_message(user_input, st.session_state['chat_history']))
                    st.markdown(f'<div class="bot-msg"><span class="avatar">{BOT_AVATAR}</span> ', unsafe_allow_html=True)
                    rendered = render_property_listings(response)
                    if not rendered:
                        st.warning("Sorry, I couldn't find any listings or tips for your query.")
                        st.markdown(f'{response}<span class="timestamp">{datetime.now().strftime("%H:%M")}</span>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.session_state['chat_history'].append({'role': 'assistant', 'content': response, 'timestamp': datetime.now().strftime("%H:%M")})
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error. Please try rephrasing your question or ask about another city."
                    st.markdown(f'<div class="error-msg">{error_msg}</div>', unsafe_allow_html=True)
                    st.session_state['chat_history'].append({'role': 'error', 'content': error_msg, 'timestamp': datetime.now().strftime("%H:%M")})

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ResourceWarning)
    main() 
