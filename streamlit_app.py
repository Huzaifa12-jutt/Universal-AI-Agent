"""
Universal AI Agent — Streamlit Frontend (Standalone)
====================================================
No separate FastAPI backend needed. Everything runs inside Streamlit:
- PDF RAG (Groq + FAISS)
- Calculator, Weather, Wikipedia, Web Search
- Gemini AI Chat
"""

import base64
import os
import tempfile
from datetime import datetime
from typing import Generator

import requests
import streamlit as st

# =========================================================================
# LOCAL BACKEND IMPORTS (Direct RAG access)
# =========================================================================
from app.rag.manager import rag_manager
from app.tools.registry import TOOLS
from app.config import settings

# =========================================================================
# CONFIG
# =========================================================================
REQUEST_TIMEOUT = 30
STREAM_TIMEOUT = 120

TOOL_META = {
    "llm":        {"emoji": "🤖", "label": "AI Chat",   "verb": "Thinking"},
    "pdf":        {"emoji": "📄", "label": "PDF RAG",   "verb": "Reading your documents"},
    "calculator": {"emoji": "🧮", "label": "Calculator", "verb": "Calculating"},
    "weather":    {"emoji": "🌤️", "label": "Weather",   "verb": "Checking the sky"},
    "wikipedia":  {"emoji": "📖", "label": "Wikipedia",  "verb": "Looking it up"},
    "search":     {"emoji": "🔎", "label": "Web Search", "verb": "Searching the web"},
    "error":      {"emoji": "⚠️", "label": "Error",      "verb": "Uh oh"},
}

SUGGESTED_PROMPTS = [
    "What's 2456 * 37 + 18?",
    "Weather in Islamabad today",
    "Who is Alan Turing?",
    "Latest news on AI agents",
]

st.set_page_config(
    page_title="Universal AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# SESSION STATE
# =========================================================================
_DEFAULTS = {
    "messages": [],
    "nav": "chat",
    "theme": "dark",
    "pending_message": None,
    "pdf_files_cache": [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# =========================================================================
# THEME (custom CSS re-skins the whole app regardless of Streamlit's own theme)
# =========================================================================
_SHARED_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    .app-title {
        font-size: 2rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(90deg, #7C5CFC, #22D3EE);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .app-subtitle { opacity: 0.65; font-size: 0.9rem; margin-top: -6px; margin-bottom: 1.2rem; }

    .tool-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 3px 12px; border-radius: 999px; font-size: 0.75rem;
        font-weight: 600; margin-bottom: 6px; border: 1px solid;
    }
    .copy-btn {
        margin-top: 6px; font-size: 0.72rem; padding: 3px 10px; border-radius: 8px;
        border: 1px solid rgba(128,128,128,0.35); background: transparent; cursor: pointer;
        opacity: 0.75; transition: all .15s ease;
    }
    .copy-btn:hover { opacity: 1; border-color: #7C5CFC; color: #7C5CFC; }

    .source-chip {
        display: inline-block; font-size: 0.72rem; padding: 2px 10px; margin: 3px 4px 0 0;
        border-radius: 999px; border: 1px solid rgba(124,92,252,0.4);
        background: rgba(124,92,252,0.08);
    }

    .pdf-card {
        border: 1px solid rgba(128,128,128,0.25); border-radius: 12px;
        padding: 10px 14px; margin-bottom: 8px; display: flex;
        justify-content: space-between; align-items: center;
    }
    .pdf-card .name { font-weight: 600; font-size: 0.88rem; }
    .pdf-card .meta { font-size: 0.72rem; opacity: 0.6; }

    .status-pill {
        display: inline-flex; align-items: center; gap: 6px; font-size: 0.75rem;
        padding: 4px 10px; border-radius: 999px; font-weight: 600;
    }
    .status-online { background: rgba(34,197,94,0.12); color: #22C55E; border: 1px solid rgba(34,197,94,0.35); }
    .status-offline { background: rgba(239,68,68,0.12); color: #EF4444; border: 1px solid rgba(239,68,68,0.35); }

    .empty-state { text-align: center; padding: 3rem 1rem; opacity: 0.7; }
    .empty-state .big { font-size: 2.4rem; }

    div[data-testid="stChatInput"] textarea { border-radius: 14px !important; }
    .stButton > button { border-radius: 10px; }
</style>
"""

_DARK_CSS = """
<style>
    .stApp { background: radial-gradient(circle at top left, #1a1530, #0E1117 55%); color: #F2F2F7; }
    section[data-testid="stSidebar"] { background: #12131c; border-right: 1px solid rgba(255,255,255,0.06); }
    .tool-badge { background: rgba(124,92,252,0.12); border-color: rgba(124,92,252,0.35); color: #B9A6FF; }
    .pdf-card { background: rgba(255,255,255,0.03); }
    .empty-state { color: #cfcfe0; }
</style>
"""

_LIGHT_CSS = """
<style>
    .stApp { background: linear-gradient(180deg, #F7F7FB, #FFFFFF 40%); color: #1A1A2E; }
    section[data-testid="stSidebar"] { background: #F2F1FA; border-right: 1px solid rgba(0,0,0,0.06); }
    .tool-badge { background: rgba(124,92,252,0.10); border-color: rgba(124,92,252,0.3); color: #5B3FD9; }
    .pdf-card { background: rgba(0,0,0,0.02); }
    .empty-state { color: #40405c; }
</style>
"""

st.markdown(_SHARED_CSS, unsafe_allow_html=True)
st.markdown(_DARK_CSS if st.session_state.theme == "dark" else _LIGHT_CSS, unsafe_allow_html=True)

# =========================================================================
# LOCAL RAG HELPERS (No API calls!)
# =========================================================================

def get_pdf_list_local() -> list:
    """Get list of uploaded PDFs directly from rag_manager"""
    files = rag_manager.list_pdfs()
    st.session_state.pdf_files_cache = files
    return files

def upload_pdfs_local(uploaded_files) -> dict:
    """Upload PDFs directly to rag_manager"""
    uploads = []
    for f in uploaded_files:
        uploads.append((f.name, f.getvalue()))
    return rag_manager.add_pdfs(uploads)

def delete_pdf_local(filename: str) -> dict:
    """Delete PDF directly from rag_manager"""
    ok = rag_manager.remove_pdf(filename)
    if ok:
        return {"status": "deleted", "filename": filename}
    return {"error": f"'{filename}' not found"}

def rebuild_index_local() -> dict:
    """Rebuild vector index directly"""
    return rag_manager.rebuild()

def clear_pdfs_local() -> dict:
    """Clear all PDFs directly"""
    rag_manager.clear_all()
    return {"status": "cleared"}

def local_chat(message: str) -> dict:
    """
    Direct chat function - decides which tool to use and returns response.
    Handles all tools: llm, pdf, calculator, weather, wikipedia, search.
    """
    msg_lower = message.lower()
    
    # Check for PDF-related queries first (if documents exist)
    if rag_manager.has_documents():
        # Check if it's a PDF-related query
        pdf_keywords = ["pdf", "document", "upload", "file", "page", "chapter", "section"]
        if any(kw in msg_lower for kw in pdf_keywords) or rag_manager.files:
            # Let RAG manager handle it
            result = rag_manager.ask(message)
            return {
                "tool": "pdf",
                "response": result.get("answer", "No response"),
                "sources": result.get("sources", [])
            }
    
    # Check for calculator
    calc_keywords = ["+", "-", "*", "×", "÷", "/", "calculate", "what is", "="]
    if any(kw in msg_lower for kw in calc_keywords):
        try:
            # Try to parse as calculator query
            import re
            # Simple math expression extraction
            nums = re.findall(r"[-+]?\d*\.?\d+", message)
            ops = re.findall(r"[+\-*/×÷]", message)
            if nums and ops:
                num1 = float(nums[0])
                num2 = float(nums[1]) if len(nums) > 1 else 0
                op = ops[0] if ops else "+"
                op_map = {"+": "add", "-": "subtract", "*": "multiply", "×": "multiply", "/": "divide", "÷": "divide"}
                op = op_map.get(op, "add")
                result = TOOLS["calculator"].run(num1=num1, num2=num2, operation=op)
                return {"tool": "calculator", "response": result, "sources": []}
        except Exception:
            pass
    
    # Check for weather
    if "weather" in msg_lower or "temperature" in msg_lower or "rain" in msg_lower:
        try:
            # Extract city name
            import re
            city_match = re.search(r"weather in (\w+)", msg_lower)
            if city_match:
                city = city_match.group(1)
                result = TOOLS["weather"].run(city)
                return {"tool": "weather", "response": result, "sources": []}
        except Exception:
            pass
    
    # Check for Wikipedia
    if "wikipedia" in msg_lower or "who is" in msg_lower or "what is" in msg_lower:
        try:
            if "wikipedia" in msg_lower:
                query = msg_lower.replace("wikipedia", "").strip()
            elif "who is" in msg_lower:
                query = msg_lower.split("who is")[-1].strip()
            elif "what is" in msg_lower:
                query = msg_lower.split("what is")[-1].strip()
            else:
                query = msg_lower
            if query:
                result = TOOLS["wikipedia"].run(query)
                return {"tool": "wikipedia", "response": result, "sources": []}
        except Exception:
            pass
    
    # Check for search
    if "search" in msg_lower or "find" in msg_lower or "google" in msg_lower:
        try:
            search_terms = msg_lower.replace("search", "").replace("find", "").replace("google", "").strip()
            if search_terms:
                result = TOOLS["search"].run(search_terms)
                return {"tool": "search", "response": result, "sources": []}
        except Exception:
            pass
    
    # Default: Use Gemini LLM (via agent)
    from app.agent import agent
    try:
        result = agent.run(message)
        return {"tool": result.get("tool", "llm"), "response": result.get("response", result.get("answer", "No response")), "sources": []}
    except Exception as e:
        return {"tool": "error", "response": f"Error: {str(e)}", "sources": []}

def local_chat_stream(message: str) -> Generator:
    """
    Stream response for chat. Yields events similar to the API format.
    """
    # Try RAG first if documents exist and query seems PDF-related
    if rag_manager.has_documents():
        pdf_keywords = ["pdf", "document", "upload", "file", "page", "chapter", "section", "summarize", "summary", "overview"]
        if any(kw in message.lower() for kw in pdf_keywords):
            # Use RAG manager with streaming
            yield {"type": "meta", "tool": "pdf"}
            # Get full answer first (RAG doesn't support streaming yet)
            result = rag_manager.ask(message)
            yield {"type": "chunk", "text": result.get("answer", "No response")}
            yield {"type": "sources", "sources": result.get("sources", [])}
            return
    
    # Otherwise use the agent
    from app.agent import agent
    try:
        # Check if agent supports streaming
        if hasattr(agent, 'stream'):
            for event in agent.stream(message):
                yield event
        else:
            # Fallback: non-streaming
            result = agent.run(message)
            yield {"type": "meta", "tool": result.get("tool", "llm")}
            response_text = result.get("response", result.get("answer", "No response"))
            yield {"type": "chunk", "text": response_text}
            yield {"type": "sources", "sources": []}
    except Exception as e:
        yield {"type": "error", "message": str(e)}

# =========================================================================
# RENDER HELPERS
# =========================================================================
def format_tool_response(tool: str, response) -> str:
    if isinstance(response, dict) and "error" in response:
        return f"❌ {response['error']}"

    if tool == "calculator" and isinstance(response, dict):
        try:
            return f"**{response['num1']:g} {response['operation']} {response['num2']:g} = {response['result']:g}**"
        except (KeyError, TypeError, ValueError):
            return str(response)

    if tool == "weather" and isinstance(response, dict):
        return (
            f"**{response.get('city', 'Unknown')}** — {response.get('condition', 'N/A')}\n\n"
            f"🌡️ {response.get('temperature_c', '?')}°C · "
            f"💧 Humidity {response.get('humidity', '?')}% · "
            f"💨 Wind {response.get('wind_kph', '?')} kph"
        )

    if tool == "wikipedia" and isinstance(response, dict):
        summary = response.get("summary", "")
        url = response.get("url", "")
        title = response.get("title", "")
        text = f"**{title}**\n\n{summary}"
        if url:
            text += f"\n\n🔗 [Read more]({url})"
        return text

    if tool == "search" and isinstance(response, dict):
        results = response.get("results", [])
        if not results:
            return "No results found."
        lines = [f"**{r.get('title', 'Untitled')}**\n{r.get('snippet', '')}\n🔗 {r.get('url', '')}" for r in results[:5]]
        return "\n\n".join(lines)

    if tool == "pdf" and isinstance(response, dict):
        return response.get("answer", str(response))

    return str(response)

def render_sources(sources: list):
    if not sources:
        return
    chips = "".join(
        f'<span class="source-chip">📄 {s.get("file", "?")} · p.{s.get("page", "?")}</span>'
        for s in sources
    )
    st.markdown(chips, unsafe_allow_html=True)

def tool_badge_html(tool: str) -> str:
    meta = TOOL_META.get(tool, TOOL_META["llm"])
    return f'<span class="tool-badge">{meta["emoji"]} {meta["label"]}</span>'

def copy_button(text: str, key: str):
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    st.markdown(
        f'''<button class="copy-btn" id="{key}" onclick="
            const bytes = Uint8Array.from(atob('{b64}'), c => c.charCodeAt(0));
            const decoded = new TextDecoder().decode(bytes);
            navigator.clipboard.writeText(decoded);
            this.innerText = '✅ Copied';
            setTimeout(() => {{ this.innerText = '📋 Copy'; }}, 1400);
        ">📋 Copy</button>''',
        unsafe_allow_html=True,
    )

def make_text_stream(events_iter, meta_box: dict):
    """Adapts the event generator into a plain text generator for st.write_stream()"""
    for event in events_iter:
        etype = event.get("type")
        if etype == "meta":
            meta_box["tool"] = event.get("tool", "llm")
        elif etype == "chunk":
            yield event.get("text", "")
        elif etype == "sources":
            meta_box["sources"] = event.get("sources", [])
        elif etype == "result":
            meta_box["tool"] = event.get("tool", meta_box.get("tool", "llm"))
            yield format_tool_response(meta_box["tool"], event.get("response"))
        elif etype == "error":
            meta_box["tool"] = "error"
            yield f"⚠️ {event.get('message', 'Something went wrong.')}"

def render_message(msg: dict, index: int):
    role = msg["role"]
    with st.chat_message("user" if role == "user" else "assistant"):
        if role == "assistant" and msg.get("tool"):
            st.markdown(tool_badge_html(msg["tool"]), unsafe_allow_html=True)
        st.markdown(msg["content"])
        if role == "assistant":
            render_sources(msg.get("sources", []))
            copy_button(msg["content"], key=f"copy_{index}")

# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown('<div class="app-title">🤖 Universal AI Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Chat · PDF RAG · Calculator · Weather · Wikipedia · Search</div>', unsafe_allow_html=True)

    # Status: Always online (since we're using local functions)
    st.markdown('<span class="status-pill status-online">🟢 Ready (Local)</span>', unsafe_allow_html=True)
    st.write("")

    nav_choice = st.radio(
        "Navigate",
        options=["chat", "documents", "settings"],
        format_func=lambda x: {"chat": "💬 AI Chat", "documents": "📄 Documents", "settings": "⚙️ Settings"}[x],
        label_visibility="collapsed",
        index=["chat", "documents", "settings"].index(st.session_state.nav),
    )
    st.session_state.nav = nav_choice

    st.divider()

    if st.session_state.nav == "chat":
        st.caption("Quick actions")
        qa1, qa2 = st.columns(2)
        with qa1:
            if st.button("🌤️ Weather", use_container_width=True):
                st.session_state.pending_message = "What's the weather in Islamabad?"
                st.rerun()
            if st.button("📖 Wikipedia", use_container_width=True):
                st.session_state.pending_message = "Who is Marie Curie?"
                st.rerun()
        with qa2:
            if st.button("🔎 Search", use_container_width=True):
                st.session_state.pending_message = "Latest news on AI agents"
                st.rerun()
            if st.button("🧮 Calculator", use_container_width=True):
                st.session_state.pending_message = "What's 245 * 18?"
                st.rerun()

        st.divider()

    pdf_files = st.session_state.pdf_files_cache or get_pdf_list_local()
    st.caption(f"📄 Documents ({len(pdf_files)})")
    if pdf_files:
        for f in pdf_files[:4]:
            st.markdown(f"— {f['name']} ({f['pages']}p)")
        if len(pdf_files) > 4:
            st.caption(f"...and {len(pdf_files) - 4} more")
    else:
        st.caption("No PDFs uploaded yet.")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with c2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.toast("Chat cleared", icon="🗑️")
            st.rerun()

    if st.session_state.messages:
        transcript = "\n\n".join(
            f"{'You' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in st.session_state.messages
        )
        st.download_button(
            "⬇️ Download Conversation",
            data=transcript,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# =========================================================================
# VIEW: CHAT
# =========================================================================
def render_chat_view():
    if not st.session_state.messages:
        st.markdown(
            f"""
            <div class="empty-state">
                <div class="big">🤖</div>
                <h3>Ask me anything</h3>
                <p>I can chat, do math, check the weather, look things up on Wikipedia,
                search the web, or answer questions about PDFs you upload in the
                <b>Documents</b> tab.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(len(SUGGESTED_PROMPTS))
        for col, prompt in zip(cols, SUGGESTED_PROMPTS):
            with col:
                if st.button(prompt, use_container_width=True, key=f"suggest_{prompt}"):
                    st.session_state.pending_message = prompt
                    st.rerun()
    else:
        for i, msg in enumerate(st.session_state.messages):
            render_message(msg, i)

    user_input = st.chat_input("Message the AI Utility Agent...")
    final_input = st.session_state.pop("pending_message", None) or user_input

    if final_input:
        st.session_state.messages.append({"role": "user", "content": final_input})
        with st.chat_message("user"):
            st.markdown(final_input)

        with st.chat_message("assistant"):
            badge_slot = st.empty()
            meta_box: dict = {}
            with st.spinner("Working on it..."):
                full_text = st.write_stream(make_text_stream(local_chat_stream(final_input), meta_box))
            tool = meta_box.get("tool", "llm")
            badge_slot.markdown(tool_badge_html(tool), unsafe_allow_html=True)
            render_sources(meta_box.get("sources", []))
            copy_button(full_text, key=f"copy_live_{len(st.session_state.messages)}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_text,
                "tool": tool,
                "sources": meta_box.get("sources", []),
            }
        )
        st.rerun()

# =========================================================================
# VIEW: DOCUMENTS (PDF RAG management)
# =========================================================================
def render_documents_view():
    st.markdown('<div class="app-title" style="font-size:1.5rem;">📄 Documents</div>', unsafe_allow_html=True)
    st.caption("Upload PDFs, then just ask about them in the AI Chat tab (e.g. \"summarize this document\").")

    uploaded = st.file_uploader(
        "Drag & drop PDFs here, or click to browse",
        type=["pdf"],
        accept_multiple_files=True,
    )

    upload_col, rebuild_col, clear_col = st.columns(3)

    with upload_col:
        if st.button("⬆️ Process & Index", use_container_width=True, type="primary", disabled=not uploaded):
            with st.status("Processing PDFs...", expanded=True) as status:
                st.write(f"📤 Uploading {len(uploaded)} file(s)...")
                result = upload_pdfs_local(uploaded)
                if "error" in result:
                    status.update(label="Upload failed", state="error")
                    st.error(result["error"])
                    st.toast("PDF upload failed", icon="❌")
                else:
                    added = result.get("added", [])
                    skipped = result.get("skipped", [])
                    st.write(f"✅ Indexed {result.get('chunks', 0)} chunk(s) from {len(added)} file(s)")
                    if skipped:
                        st.write(f"⏭️ Skipped (already uploaded): {', '.join(skipped)}")
                    status.update(label="Documents ready!", state="complete")
                    st.toast(f"{len(added)} PDF(s) indexed", icon="✅")
            get_pdf_list_local()
            st.rerun()

    with rebuild_col:
        if st.button("🔄 Rebuild Index", use_container_width=True, disabled=not st.session_state.pdf_files_cache):
            with st.spinner("Rebuilding vector index..."):
                result = rebuild_index_local()
            if "error" in result:
                st.error(result["error"])
                st.toast("Rebuild failed", icon="❌")
            else:
                st.toast("Index rebuilt", icon="🔄")
            st.rerun()

    with clear_col:
        if st.button("🧹 Clear All PDFs", use_container_width=True, disabled=not st.session_state.pdf_files_cache):
            clear_pdfs_local()
            st.session_state.pdf_files_cache = []
            st.toast("All PDFs removed", icon="🧹")
            st.rerun()

    st.divider()

    pdf_files = get_pdf_list_local()
    st.subheader(f"Current PDFs ({len(pdf_files)})")

    if not pdf_files:
        st.info("No PDFs uploaded yet. Add some above to enable PDF chat.")
        return

    for f in pdf_files:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f"""<div class="pdf-card">
                        <div>
                            <div class="name">📄 {f['name']}</div>
                            <div class="meta">{f['pages']} page(s) · {f['size_kb']} KB</div>
                        </div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("🗑️", key=f"del_{f['name']}", use_container_width=True):
                res = delete_pdf_local(f["name"])
                if "error" in res:
                    st.toast(res["error"], icon="❌")
                else:
                    st.toast(f"Removed {f['name']}", icon="🗑️")
                st.rerun()

# =========================================================================
# VIEW: SETTINGS
# =========================================================================
def render_settings_view():
    st.markdown('<div class="app-title" style="font-size:1.5rem;">⚙️ Settings</div>', unsafe_allow_html=True)

    st.subheader("Appearance")
    theme_choice = st.radio("Theme", options=["dark", "light"], index=0 if st.session_state.theme == "dark" else 1, horizontal=True)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.divider()

    st.subheader("System Status")
    st.info("✅ Running in standalone mode (no external backend required)")
    
    st.subheader("PDF RAG Status")
    files = rag_manager.list_pdfs()
    if files:
        st.success(f"✅ {len(files)} PDF(s) loaded")
        for f in files:
            st.write(f"• {f['name']} ({f['pages']} pages)")
    else:
        st.warning("No PDFs loaded. Upload some in the Documents tab.")

    st.divider()

    st.subheader("Conversation")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.toast("Chat cleared", icon="🗑️")
            st.rerun()
    with c2:
        if st.session_state.messages:
            transcript = "\n\n".join(
                f"{'You' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "⬇️ Download Conversation",
                data=transcript,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.divider()

    with st.expander("ℹ️ About this app"):
        st.markdown(
            """
            **Universal AI Agent** is a single Streamlit app that routes every message
            to the right tool automatically:

            - 🤖 **AI Chat** - powered by Gemini
            - 📄 **PDF RAG** - LangChain + FastEmbed + FAISS + Groq, over PDFs you upload
            - 🧮 **Calculator**, 🌤️ **Weather**, 📖 **Wikipedia**, 🔎 **Web Search**

            All tools run locally within the Streamlit app — no separate backend needed.
            """
        )

# =========================================================================
# ROUTER
# =========================================================================
if st.session_state.nav == "chat":
    render_chat_view()
elif st.session_state.nav == "documents":
    render_documents_view()
else:
    render_settings_view()
