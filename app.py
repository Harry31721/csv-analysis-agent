import os
import sys
import tempfile
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # works locally via .env

# Streamlit Cloud: copy secrets into os.environ so langchain-groq can find them
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(str(_k), str(_v))
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph import build_graph
from utils.code_executor import clear_figures, get_figures

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analysis Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  (fine-tuning on top of config.toml dark theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Core backgrounds ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background: #0f172a !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: #080d1a !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #64748b !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e2d4a !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(100,116,139,0.2) !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-size: 0.81em !important;
    text-align: left !important;
    transition: all 0.18s ease !important;
    padding: 7px 12px !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.5) !important;
    color: #a5b4fc !important;
    transform: translateX(4px) !important;
}

/* File uploader */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(99,102,241,0.35) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(99,102,241,0.65) !important;
    background: rgba(99,102,241,0.05) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #475569 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #131c35 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
    margin-bottom: 12px !important;
    padding: 6px 10px !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #0d1424 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    margin: 5px 0 !important;
}
[data-testid="stExpander"] details summary {
    font-size: 0.83em !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    color: #818cf8 !important;
}
[data-testid="stExpander"] details summary:hover {
    color: #a5b4fc !important;
    background: rgba(99,102,241,0.06) !important;
}

/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    background: #131c35 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}
[data-testid="stStatusWidget"] p { color: #94a3b8 !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #131c35 !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-size: 0.95em !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #334155 !important; }

/* ── Code blocks ── */
.stCode, [data-testid="stCode"], pre {
    background: #080d1a !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 10px !important;
}

/* ── Images (charts) ── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
    margin: 8px 0 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── Dividers ── */
hr { border-color: #1e2d4a !important; margin: 10px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "file_path" not in st.session_state:
    st.session_state.file_path: str | None = None
if "last_filename" not in st.session_state:
    st.session_state.last_filename: str | None = None
if "file_stats" not in st.session_state:
    st.session_state.file_stats: dict | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def reset_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    clear_figures()


def load_file_stats(path: str) -> dict:
    try:
        df = pd.read_csv(path)
        size_kb = os.path.getsize(path) / 1024
        return {"rows": df.shape[0], "cols": df.shape[1], "size": f"{size_kb:.1f} KB"}
    except Exception:
        return {}


def extract_text(content) -> str:
    """Safely pull a plain string out of any LangChain message content."""
    if not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()
    return str(content).strip()


# ── Tool display metadata ─────────────────────────────────────────────────────
_TOOL_META: dict[str, tuple[str, str, str]] = {
    "read_csv_file":       ("📄", "#38bdf8", "Read CSV"),
    "analyze_dataframe":   ("📊", "#a78bfa", "Analyse Data"),
    "execute_python_code": ("🐍", "#34d399", "Execute Code"),
}


def render_reasoning_chain(tool_steps: list[dict]) -> None:
    """Render a visual timeline of every tool the agent called."""
    if not tool_steps:
        return

    # Build every node as a single-line string — multi-line f-strings add
    # leading spaces which Markdown misreads as code blocks.
    nodes_html = ""
    for i, step in enumerate(tool_steps):
        tname = step["tool"]
        icon, color, label = _TOOL_META.get(tname, ("🔧", "#818cf8", tname))
        is_last = (i == len(tool_steps) - 1)

        connector = "" if is_last else (
            f'<div style="width:2px;height:14px;'
            f'background:linear-gradient({color}66,{color}11);'
            f'margin:2px 0 2px 11px;border-radius:2px;"></div>'
        )

        dot = (
            f'<div style="width:24px;height:24px;border-radius:50%;'
            f'background:{color}18;border:1.5px solid {color};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:0.7em;">{icon}</div>'
        )
        text = (
            f'<span style="color:{color};font-size:0.82em;font-weight:600;">{label}</span>'
            f'<span style="color:#334155;font-size:0.75em;margin-left:6px;'
            f'font-family:monospace;">{tname}</span>'
        )
        row = (
            f'<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:2px;">'
            f'<div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;">{dot}</div>'
            f'<div style="padding-top:3px;">{text}</div>'
            f'</div>'
        )
        nodes_html += row + connector

    header = (
        '<div style="font-size:0.68em;font-weight:700;color:#334155;'
        'text-transform:uppercase;letter-spacing:1.2px;margin-bottom:12px;">'
        '&#9881; &nbsp;Agent Reasoning Chain</div>'
    )
    wrapper = (
        f'<div style="background:#080d1a;border:1px solid rgba(99,102,241,0.18);'
        f'border-radius:12px;padding:14px 16px;margin:8px 0 12px 0;">'
        f'{header}{nodes_html}</div>'
    )
    st.markdown(wrapper, unsafe_allow_html=True)

    # Expandable details per step
    for step in tool_steps:
        tname = step["tool"]
        icon, color, label = _TOOL_META.get(tname, ("🔧", "#818cf8", tname))
        if step.get("code") or step.get("output"):
            with st.expander(f"{icon}  {label}  —  details", expanded=False):
                if step.get("code"):
                    st.code(step["code"], language="python")
                raw = step.get("output", "")
                if raw:
                    preview = raw[:900] + ("…" if len(raw) > 900 else "")
                    st.text(preview)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Branding
    st.markdown("""
    <div style="padding:14px 6px 18px 6px;">
        <div style="font-size:1.4rem;font-weight:700;color:#e2e8f0;letter-spacing:-0.3px;">
            🤖 Data Agent
        </div>
        <div style="margin-top:4px;display:flex;gap:5px;flex-wrap:wrap;">
            <span style="background:rgba(129,140,248,0.12);color:#818cf8;padding:2px 7px;
                         border-radius:6px;font-size:0.68em;font-weight:600;border:1px solid rgba(129,140,248,0.25);">
                Llama 3.3</span>
            <span style="background:rgba(56,189,248,0.1);color:#38bdf8;padding:2px 7px;
                         border-radius:6px;font-size:0.68em;font-weight:600;border:1px solid rgba(56,189,248,0.2);">
                Groq</span>
            <span style="background:rgba(52,211,153,0.1);color:#34d399;padding:2px 7px;
                         border-radius:6px;font-size:0.68em;font-weight:600;border:1px solid rgba(52,211,153,0.2);">
                LangGraph</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    # Upload
    st.markdown(
        '<p style="font-size:0.72em;font-weight:700;color:#334155;'
        'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">'
        'Upload Dataset</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    if uploaded and uploaded.name != st.session_state.last_filename:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
            tmp.write(uploaded.getvalue())
            st.session_state.file_path = tmp.name
        st.session_state.last_filename = uploaded.name
        st.session_state.file_stats = load_file_stats(st.session_state.file_path)
        reset_conversation()

    sample_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sample_data", "sales_data_messy.csv"
    )
    if os.path.exists(sample_path):
        if st.button("📂  Use sample dataset", use_container_width=True):
            if st.session_state.file_path != sample_path:
                st.session_state.file_path = sample_path
                st.session_state.last_filename = "sales_data_messy.csv"
                st.session_state.file_stats = load_file_stats(sample_path)
                reset_conversation()
            st.rerun()

    # File stats card
    if st.session_state.file_path and st.session_state.file_stats:
        stats = st.session_state.file_stats
        fname = st.session_state.last_filename or "file.csv"
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.22);
                    border-radius:10px;padding:12px 14px;margin:14px 0 4px 0;">
            <div style="color:#475569;font-size:0.68em;font-weight:700;
                        letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">
                📁 Active File
            </div>
            <div style="color:#cbd5e1;font-size:0.85em;font-weight:600;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                {fname}
            </div>
            <div style="display:flex;gap:20px;margin-top:10px;">
                <div>
                    <div style="color:#818cf8;font-size:1em;font-weight:700;">
                        {stats.get("rows","—")}</div>
                    <div style="color:#334155;font-size:0.66em;margin-top:1px;">rows</div>
                </div>
                <div>
                    <div style="color:#818cf8;font-size:1em;font-weight:700;">
                        {stats.get("cols","—")}</div>
                    <div style="color:#334155;font-size:0.66em;margin-top:1px;">columns</div>
                </div>
                <div>
                    <div style="color:#818cf8;font-size:1em;font-weight:700;">
                        {stats.get("size","—")}</div>
                    <div style="color:#334155;font-size:0.66em;margin-top:1px;">size</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Example questions
    if st.session_state.file_path:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.72em;font-weight:700;color:#334155;'
            'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Try asking</p>',
            unsafe_allow_html=True,
        )
        for ex in [
            "What does this dataset look like?",
            "Find all data quality issues",
            "Show a bar chart of sales by region",
            "Detect outliers in numeric columns",
            "Write Python code to clean this data",
        ]:
            if st.button(ex, use_container_width=True, key=f"ex_{hash(ex)}"):
                st.session_state.pending_input = ex
                st.rerun()

    # Clear + footer
    st.markdown('<hr>', unsafe_allow_html=True)
    if st.button("🗑️  Clear conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.markdown("""
    <div style="margin-top:24px;text-align:center;font-size:0.66em;color:#1e2d4a;padding-bottom:6px;">
        Built with LangGraph · Groq · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#312e81 0%,#1e3a5f 50%,#0c2340 100%);
            border:1px solid rgba(129,140,248,0.25);
            border-radius:16px;padding:26px 30px 22px 30px;margin-bottom:22px;
            box-shadow:0 4px 32px rgba(99,102,241,0.15),inset 0 1px 0 rgba(255,255,255,0.05);">
    <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.4px;">
        🤖 AI Data Analysis Agent
    </div>
    <div style="color:rgba(203,213,225,0.6);margin-top:6px;font-size:0.86rem;line-height:1.5;">
        Upload any CSV &nbsp;·&nbsp; Ask questions in plain English &nbsp;·&nbsp;
        Agent reasons step-by-step, writes &amp; runs Python automatically
    </div>
    <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap;">
        <span style="background:rgba(129,140,248,0.15);color:#a5b4fc;padding:4px 12px;
                     border-radius:20px;font-size:0.71em;font-weight:600;
                     border:1px solid rgba(129,140,248,0.3);">⚡ Llama 3.3 70B</span>
        <span style="background:rgba(56,189,248,0.1);color:#7dd3fc;padding:4px 12px;
                     border-radius:20px;font-size:0.71em;font-weight:600;
                     border:1px solid rgba(56,189,248,0.25);">🔗 LangGraph ReAct</span>
        <span style="background:rgba(52,211,153,0.1);color:#6ee7b7;padding:4px 12px;
                     border-radius:20px;font-size:0.71em;font-weight:600;
                     border:1px solid rgba(52,211,153,0.25);">🛠️ Tool Calling</span>
        <span style="background:rgba(251,146,60,0.1);color:#fbbf24;padding:4px 12px;
                     border-radius:20px;font-size:0.71em;font-weight:600;
                     border:1px solid rgba(251,146,60,0.25);">🐍 Code Execution</span>
        <span style="background:rgba(167,139,250,0.1);color:#c4b5fd;padding:4px 12px;
                     border-radius:20px;font-size:0.71em;font-weight:600;
                     border:1px solid rgba(167,139,250,0.25);">💾 Conversation Memory</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Landing page  (no file loaded)
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.file_path:
    st.markdown("""
    <div style="text-align:center;padding:24px 0 16px 0;">
        <div style="font-size:1rem;color:#475569;font-weight:500;">
            Upload a CSV file or use the sample dataset to get started
        </div>
        <div style="font-size:0.83rem;color:#334155;margin-top:5px;">
            The agent will read your data, identify issues, generate code, and produce visualisations
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")
    cards = [
        ("📊", "#38bdf8", "Smart Analysis",
         "Profiles your dataset automatically — detects nulls, duplicates, type mismatches, and mixed formats."),
        ("🔍", "#a78bfa", "Anomaly Detection",
         "Finds statistical outliers, negative values, inconsistent casing, and data integrity violations."),
        ("🐍", "#34d399", "Code Generation",
         "Writes and executes real pandas & matplotlib code. Returns charts and a clean, downloadable dataset."),
    ]
    for col, (icon, color, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="background:#131c35;border:1px solid rgba(255,255,255,0.06);
                        border-radius:16px;padding:26px 20px;text-align:center;
                        box-shadow:0 2px 16px rgba(0,0,0,0.3);height:100%;
                        transition:transform 0.2s,box-shadow 0.2s;">
                <div style="width:48px;height:48px;border-radius:14px;margin:0 auto 14px auto;
                            background:{color}15;border:1px solid {color}30;
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.5rem;">{icon}</div>
                <div style="font-weight:700;font-size:0.96rem;color:#e2e8f0;margin-bottom:8px;">
                    {title}</div>
                <div style="font-size:0.82rem;color:#475569;line-height:1.65;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # How it works
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#080d1a;border:1px solid rgba(255,255,255,0.05);
                border-radius:14px;padding:20px 24px;">
        <div style="font-size:0.7em;font-weight:700;color:#334155;
                    text-transform:uppercase;letter-spacing:1.2px;margin-bottom:14px;">
            ⚙ How it works
        </div>
        <div style="display:flex;gap:0;flex-wrap:wrap;">
            <div style="flex:1;min-width:140px;padding:0 12px 0 0;">
                <div style="color:#818cf8;font-size:0.82em;font-weight:700;margin-bottom:4px;">
                    1 · Upload</div>
                <div style="color:#475569;font-size:0.78em;line-height:1.5;">
                    Drop any CSV — clean or messy</div>
            </div>
            <div style="color:#1e2d4a;font-size:1.2em;padding-top:4px;">›</div>
            <div style="flex:1;min-width:140px;padding:0 12px;">
                <div style="color:#38bdf8;font-size:0.82em;font-weight:700;margin-bottom:4px;">
                    2 · Ask</div>
                <div style="color:#475569;font-size:0.78em;line-height:1.5;">
                    Type a question in plain English</div>
            </div>
            <div style="color:#1e2d4a;font-size:1.2em;padding-top:4px;">›</div>
            <div style="flex:1;min-width:140px;padding:0 12px;">
                <div style="color:#34d399;font-size:0.82em;font-weight:700;margin-bottom:4px;">
                    3 · Agent Reasons</div>
                <div style="color:#475569;font-size:0.78em;line-height:1.5;">
                    Calls tools, writes &amp; runs code</div>
            </div>
            <div style="color:#1e2d4a;font-size:1.2em;padding-top:4px;">›</div>
            <div style="flex:1;min-width:140px;padding:0 0 0 12px;">
                <div style="color:#fbbf24;font-size:0.82em;font-weight:700;margin-bottom:4px;">
                    4 · Results</div>
                <div style="color:#475569;font-size:0.78em;line-height:1.5;">
                    Charts, insights &amp; clean code</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Conversation history
# ─────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_reasoning_chain(msg.get("tool_steps", []))
            st.markdown(msg["content"])
            for fig_bytes in msg.get("figures", []):
                st.image(fig_bytes, use_container_width=True)
        else:
            st.markdown(msg["content"])


# ─────────────────────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────────────────────
user_input: str | None = st.session_state.pop("pending_input", None)
if not user_input:
    user_input = st.chat_input("Ask anything about your data…")
if not user_input:
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Run agent
# ─────────────────────────────────────────────────────────────────────────────
st.session_state.messages.append({"role": "user", "content": user_input})
with st.chat_message("user"):
    st.markdown(user_input)

clear_figures()

with st.chat_message("assistant"):
    tool_steps: list[dict] = []
    ai_tool_calls: dict[str, dict] = {}
    final_answer = ""

    with st.status("🤔  Agent is reasoning…", expanded=True) as status:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        inputs = {"messages": [("human", user_input)], "file_path": st.session_state.file_path}

        for chunk in st.session_state.graph.stream(
            inputs, config=config, stream_mode="updates"
        ):
            for node_name, node_output in chunk.items():
                msgs = node_output.get("messages", [])

                if node_name == "agent":
                    for m in msgs:
                        tool_calls = getattr(m, "tool_calls", None)
                        if tool_calls:                          # calling a tool
                            for tc in tool_calls:
                                ai_tool_calls[tc["id"]] = tc
                                icon, color, label = _TOOL_META.get(
                                    tc["name"], ("🔧", "#818cf8", tc["name"])
                                )
                                st.markdown(
                                    f'<span style="color:{color};font-size:0.87em;font-weight:600;">'
                                    f'{icon}&nbsp; Calling <code style="color:{color};">'
                                    f'{tc["name"]}</code>…</span>',
                                    unsafe_allow_html=True,
                                )
                        else:                                   # final answer
                            text = extract_text(getattr(m, "content", ""))
                            if text:
                                final_answer = text

                elif node_name == "tools":
                    for m in msgs:
                        call_id = getattr(m, "tool_call_id", "")
                        call_info = ai_tool_calls.get(call_id, {})
                        tname = call_info.get("name", getattr(m, "name", "tool"))
                        targs = call_info.get("args", {})
                        tool_steps.append({
                            "tool": tname,
                            "code": targs.get("code") if tname == "execute_python_code" else None,
                            "output": str(getattr(m, "content", "")),
                        })
                        icon, color, label = _TOOL_META.get(tname, ("✓", "#34d399", tname))
                        st.markdown(
                            f'<span style="color:{color};font-size:0.82em;">'
                            f'✓&nbsp; <code style="color:{color};">{tname}</code>'
                            f'&nbsp;completed</span>',
                            unsafe_allow_html=True,
                        )

        status.update(label="✅  Analysis complete", state="complete")

    # ── Render reasoning chain, answer, charts ──
    render_reasoning_chain(tool_steps)

    if final_answer:
        st.markdown(final_answer)
    else:
        st.warning("The agent didn't return a text response. Check the tool outputs above.")

    figures = get_figures()
    for fig_bytes in figures:
        st.image(fig_bytes, use_container_width=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "tool_steps": tool_steps,
        "figures": figures,
    })
