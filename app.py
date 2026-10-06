import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from src.ai.gemini_client import GeminiClient
from src.ingestion.pipeline import KnowledgeIngestion
from src.rag.store import ChromaStore
from src.rag.retriever import Retriever


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Voice2Knowledge",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLES
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(99,102,241,0.10), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(139,92,246,0.08), transparent 25%),
        #080b14;
    color: #f8fafc;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.main .block-container {
    max-width: 1450px;
    padding: 2rem 2.5rem 4rem 2.5rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: #0b0f1a;
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 10px 25px 10px;
}

.logo-icon {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    box-shadow: 0 8px 25px rgba(99,102,241,0.35);
    font-size: 21px;
}

.logo-text {
    font-size: 19px;
    font-weight: 750;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.sidebar-section {
    color: #64748b;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    padding: 18px 12px 8px;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border: none;
    background: transparent;
    color: #94a3b8;
    text-align: left;
    border-radius: 10px;
    padding: 11px 13px;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,0.10);
    color: #ffffff;
    border: none;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 42px;
    margin-bottom: 28px;

    background:
        radial-gradient(
            circle at 80% 20%,
            rgba(139,92,246,0.30),
            transparent 32%
        ),
        radial-gradient(
            circle at 10% 100%,
            rgba(59,130,246,0.20),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #11182c,
            #15122c
        );

    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
}

.hero-label {
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}

.hero h1 {
    font-size: 42px;
    line-height: 1.12;
    letter-spacing: -1.8px;
    margin: 0 0 15px 0;
    color: #ffffff;
    max-width: 700px;
}

.hero p {
    color: #aab4c8;
    font-size: 16px;
    line-height: 1.7;
    max-width: 650px;
    margin: 0;
}


/* ============================================================
   METRICS
   ============================================================ */

.metric-card {
    background: rgba(17,24,39,0.72);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 21px;
    min-height: 115px;
    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99,102,241,0.35);
    box-shadow: 0 12px 35px rgba(0,0,0,0.20);
}

.metric-icon {
    font-size: 20px;
    margin-bottom: 12px;
}

.metric-label {
    color: #64748b;
    font-size: 12px;
    margin-bottom: 5px;
}

.metric-value {
    color: #ffffff;
    font-size: 27px;
    font-weight: 750;
    letter-spacing: -0.8px;
}


/* ============================================================
   CARDS
   ============================================================ */

.card {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.card-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 5px;
}

.card-subtitle {
    color: #64748b;
    font-size: 13px;
}

.section-title {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 30px 0 5px;
}

.section-description {
    color: #64748b;
    font-size: 13px;
    margin-bottom: 18px;
}


/* ============================================================
   UPLOAD
   ============================================================ */

.upload-box {
    border: 1.5px dashed rgba(129,140,248,0.45);
    border-radius: 18px;
    padding: 35px 25px;
    text-align: center;

    background:
        linear-gradient(
            180deg,
            rgba(99,102,241,0.07),
            rgba(99,102,241,0.02)
        );
}

.upload-icon {
    font-size: 34px;
    margin-bottom: 10px;
}

.upload-title {
    color: #f8fafc;
    font-size: 16px;
    font-weight: 650;
}

.upload-description {
    color: #64748b;
    font-size: 13px;
    margin-top: 6px;
}


/* ============================================================
   FEATURE CARDS
   ============================================================ */

.feature-card {
    height: 100%;
    padding: 25px;
    border-radius: 18px;
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.07);
}

.feature-icon {
    width: 45px;
    height: 45px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(99,102,241,0.13);
    font-size: 21px;
    margin-bottom: 18px;
}

.feature-title {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 8px;
}

.feature-description {
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
}


/* ============================================================
   KNOWLEDGE
   ============================================================ */

.knowledge-item {
    display: flex;
    align-items: center;
    gap: 15px;
    padding: 15px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.knowledge-icon {
    width: 42px;
    height: 42px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(99,102,241,0.12);
    font-size: 18px;
}

.knowledge-name {
    color: #e2e8f0;
    font-size: 14px;
    font-weight: 600;
}

.knowledge-meta {
    color: #64748b;
    font-size: 12px;
    margin-top: 3px;
}


/* ============================================================
   ACTIVITY
   ============================================================ */

.activity-item {
    display: flex;
    gap: 12px;
    padding: 13px 0;
}

.activity-dot {
    width: 8px;
    height: 8px;
    margin-top: 5px;
    border-radius: 50%;
    background: #6366f1;
    box-shadow: 0 0 12px rgba(99,102,241,0.7);
}

.activity-text {
    color: #cbd5e1;
    font-size: 13px;
    line-height: 1.5;
}

.activity-time {
    color: #475569;
    font-size: 11px;
    margin-top: 3px;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    border-radius: 11px;
    border: 1px solid rgba(255,255,255,0.08);
    background: #111827;
    color: #e2e8f0;
    font-weight: 600;
    padding: 10px 18px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #6366f1;
    background: rgba(99,102,241,0.12);
    color: #ffffff;
}

.primary-button button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.25);
}


/* ============================================================
   INPUTS
   ============================================================ */

div[data-testid="stTextInput"] input {
    background: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 1px #6366f1 !important;
}

section[data-testid="stFileUploaderDropzone"] {
    background: rgba(15,23,42,0.65);
    border: 1px dashed rgba(129,140,248,0.4);
    border-radius: 15px;
}


/* ============================================================
   CHAT
   ============================================================ */

[data-testid="stChatMessage"] {
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 15px;
}

[data-testid="stChatInput"] {
    background: #0f172a;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
}


/* ============================================================
   STATUS
   ============================================================ */

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 11px;
    border-radius: 999px;
    background: rgba(34,197,94,0.09);
    border: 1px solid rgba(34,197,94,0.18);
    color: #86efac;
    font-size: 11px;
    font-weight: 600;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 10px rgba(34,197,94,0.7);
}


/* ============================================================
   TAG
   ============================================================ */

.tag {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 7px;
    background: rgba(99,102,241,0.10);
    color: #a5b4fc;
    font-size: 10px;
    font-weight: 600;
    margin-right: 5px;
}


/* ============================================================
   CLEANUP
   ============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    visibility: hidden;
}

hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 25px 0;
}

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #080b14;
}

::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 10px;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 900px) {

    .main .block-container {
        padding: 1.2rem;
    }

    .hero {
        padding: 28px;
    }

    .hero h1 {
        font-size: 30px;
    }

    .metric-card {
        margin-bottom: 12px;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND
# ============================================================

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_DIR = DATA_DIR / "chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


try:
    gemini = GeminiClient()
    store = ChromaStore(str(DB_DIR))
    retriever = Retriever(gemini, store)
    ingestion = KnowledgeIngestion(gemini, store)

except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_count" not in st.session_state:
    st.session_state.uploaded_count = 0


def go_to(page):
    st.session_state.page = page


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">
            <div class="logo-icon">🧠</div>
            <div class="logo-text">Voice2Knowledge</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">Workspace</div>',
                unsafe_allow_html=True)

    if st.button("⌂  Dashboard", use_container_width=True):
        go_to("Dashboard")
        st.rerun()

    if st.button("📚  Knowledge Library", use_container_width=True):
        go_to("Library")
        st.rerun()

    if st.button("💬  Ask Knowledge", use_container_width=True):
        go_to("Ask")
        st.rerun()

    if st.button("🎓  Study Assistant", use_container_width=True):
        go_to("Study")
        st.rerun()

    st.markdown('<div class="sidebar-section">Tools</div>',
                unsafe_allow_html=True)

    if st.button("📝  Summaries", use_container_width=True):
        go_to("Summaries")
        st.rerun()

    if st.button("🎙️  Lecture Mode", use_container_width=True):
        go_to("Lecture")
        st.rerun()

    if st.button("⚙️  Settings", use_container_width=True):
        go_to("Settings")
        st.rerun()

    st.markdown("---")

    st.markdown(
        """
        <div class="status-badge">
            <span class="status-dot"></span>
            AI Knowledge Engine Ready
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Gemini + Chroma RAG")


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard():

    sources = store.list_sources()

    documents = len(sources)

    total_chunks = sum(
        int(item.get("chunks", 0))
        for item in sources
        if str(item.get("chunks", "0")).isdigit()
    )

    questions = len(
        [
            m
            for m in st.session_state.messages
            if m.get("role") == "user"
        ]
    )

    st.markdown(
        """
        <div class="hero">

            <div class="hero-label">
                PERSONAL KNOWLEDGE INTELLIGENCE
            </div>

            <h1>
                Turn everything you learn into knowledge you can use.
            </h1>

            <p>
                Upload your PDFs, lectures, screenshots and notes.
                Voice2Knowledge turns them into a searchable,
                intelligent knowledge base you can ask questions about.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("📄", "Documents", documents),
        ("🧩", "Knowledge Chunks", total_chunks),
        ("💬", "Questions", questions),
        ("⚡", "AI Engine", "Ready"),
    ]

    for column, (icon, label, value) in zip(
        [c1, c2, c3, c4],
        metrics,
    ):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # ADD KNOWLEDGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Add Knowledge</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Import the material you want Voice2Knowledge to understand.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-box">

            <div class="upload-icon">☁️</div>

            <div class="upload-title">
                Upload your knowledge
            </div>

            <div class="upload-description">
                PDF, text, images, markdown and audio files
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Choose files",
        type=[
            "pdf",
            "txt",
            "md",
            "png",
            "jpg",
            "jpeg",
            "mp3",
            "wav",
            "m4a",
            "webm",
        ],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:

        if st.button(
            "⚡ Process & Index Knowledge",
            use_container_width=True,
        ):

            progress = st.progress(0)

            for i, file in enumerate(uploaded):

                target = UPLOAD_DIR / file.name
                target.write_bytes(file.getbuffer())

                try:

                    count = ingestion.process_file(target)

                    st.success(
                        f"{file.name} — {count} chunks indexed"
                    )

                except Exception as e:

                    st.error(
                        f"{file.name}: {e}"
                    )

                progress.progress(
                    (i + 1) / len(uploaded)
                )

            st.rerun()

    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Quick Actions</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">Jump directly into your knowledge.</div>',
        unsafe_allow_html=True,
    )

    q1, q2, q3 = st.columns(3)

    with q1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">
                    Ask your knowledge
                </div>
                <div class="feature-description">
                    Ask questions and get answers grounded in
                    your indexed documents.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Open Ask", key="ask_home", use_container_width=True):
            go_to("Ask")
            st.rerun()

    with q2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎓</div>
                <div class="feature-title">
                    Study smarter
                </div>
                <div class="feature-description">
                    Generate revision material and quizzes from
                    your existing knowledge base.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Study",
            key="study_home",
            use_container_width=True,
        ):
            go_to("Study")
            st.rerun()

    with q3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📝</div>
                <div class="feature-title">
                    Create summaries
                </div>
                <div class="feature-description">
                    Turn your indexed material into concise
                    revision-ready summaries.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open Summaries",
            key="summary_home",
            use_container_width=True,
        ):
            go_to("Summaries")
            st.rerun()

    # --------------------------------------------------------
    # RECENT KNOWLEDGE + ACTIVITY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Your Knowledge</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.7, 1])

    with left:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    📚 Recent Knowledge
                </div>
                <div class="card-subtitle">
                    Your indexed sources
                </div>
            """,
            unsafe_allow_html=True,
        )

        if not sources:

            st.info(
                "Your library is empty. Upload your first document above."
            )

        else:

            for item in sources[:8]:

                source_name = item.get("source", "Unknown")
                source_type = item.get("type", "file")
                chunks = item.get("chunks", 0)

                icon = "📄"

                if source_type in ["png", "jpg", "jpeg"]:
                    icon = "🖼️"
                elif source_type in ["mp3", "wav", "m4a", "webm"]:
                    icon = "🎙️"
                elif source_type in ["txt", "md"]:
                    icon = "📝"

                st.markdown(
                    f"""
                    <div class="knowledge-item">

                        <div class="knowledge-icon">
                            {icon}
                        </div>

                        <div>
                            <div class="knowledge-name">
                                {source_name}
                            </div>

                            <div class="knowledge-meta">
                                {source_type.upper()} · {chunks} chunks
                            </div>
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    ⚡ Recent Activity
                </div>

                <div class="card-subtitle">
                    Your latest interactions
                </div>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.messages:

            st.markdown(
                """
                <div class="activity-item">
                    <div class="activity-dot"></div>
                    <div>
                        <div class="activity-text">
                            Your activity will appear here.
                        </div>
                        <div class="activity-time">
                            Start by asking a question.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            recent = st.session_state.messages[-5:]

            for message in reversed(recent):

                role = message.get("role")

                if role == "user":
                    text = message.get("content", "")
                    label = "Asked a question"
                else:
                    text = message.get("content", "")
                    label = "Generated an answer"

                text = text[:100]

                st.markdown(
                    f"""
                    <div class="activity-item">

                        <div class="activity-dot"></div>

                        <div>
                            <div class="activity-text">
                                <strong>{label}</strong><br>
                                {text}
                            </div>

                            <div class="activity-time">
                                Recent
                            </div>
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ASK KNOWLEDGE
# ============================================================

def render_ask():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-label">
                KNOWLEDGE SEARCH
            </div>

            <h1>
                Ask anything you have learned.
            </h1>

            <p>
                Voice2Knowledge searches your indexed knowledge
                and generates an answer using the relevant context.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sources"):

                with st.expander("📌 Sources"):

                    for source in message["sources"]:
                        st.markdown(source)

    question = st.chat_input(
        "Ask something from your knowledge..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching your knowledge base..."
            ):

                try:
                    answer, sources = retriever.answer(question)

                except Exception as e:

                    answer = (
                        "I couldn't complete the search right now."
                    )

                    sources = []

                    st.error(str(e))

            st.markdown(answer)

            if sources:

                with st.expander("📌 Sources"):

                    for source in sources:
                        st.markdown(source)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )


# ============================================================
# LIBRARY
# ============================================================

def render_library():

    st.markdown(
        '<div class="section-title">Knowledge Library</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Everything currently indexed by Voice2Knowledge.
        </div>
        """,
        unsafe_allow_html=True,
    )

    sources = store.list_sources()

    if not sources:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    Your library is empty
                </div>
                <div class="card-subtitle">
                    Upload a PDF, lecture, note or image to begin.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    for item in sources:

        source_name = item.get("source", "Unknown")
        source_type = item.get("type", "file")
        chunks = item.get("chunks", 0)

        col1, col2 = st.columns([4, 1])

        with col1:

            st.markdown(
                f"""
                <div class="card">

                    <div class="knowledge-item">

                        <div class="knowledge-icon">
                            📄
                        </div>

                        <div>

                            <div class="knowledge-name">
                                {source_name}
                            </div>

                            <div class="knowledge-meta">
                                {source_type.upper()} ·
                                {chunks} chunks
                            </div>

                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.metric(
                "Chunks",
                chunks,
            )


# ============================================================
# STUDY
# ============================================================

def render_study():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-label">
                STUDY ASSISTANT
            </div>

            <h1>
                Learn from your own knowledge base.
            </h1>

            <p>
                Generate revision material and questions using
                the information you have already indexed.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">📝</div>

                <div class="feature-title">
                    Revision Summary
                </div>

                <div class="feature-description">
                    Generate a concise study summary from your
                    indexed knowledge.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Generate Revision Summary",
            use_container_width=True,
        ):

            with st.spinner("Generating revision material..."):

                result = retriever.study_action(
                    """
                    Create a concise revision summary with:
                    - clear headings
                    - important concepts
                    - bullet points
                    - key facts

                    Use only the supplied knowledge context.
                    """
                )

            st.markdown("---")
            st.markdown(result)

    with c2:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">❓</div>

                <div class="feature-title">
                    Generate Quiz
                </div>

                <div class="feature-description">
                    Create study questions from your indexed
                    documents.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Generate Quiz",
            use_container_width=True,
        ):

            with st.spinner("Creating quiz..."):

                result = retriever.study_action(
                    """
                    Create 10 study questions based only on
                    the supplied context.

                    Include a mixture of:
                    - conceptual questions
                    - factual questions
                    - application questions
                    """
                )

            st.markdown("---")
            st.markdown(result)


# ============================================================
# SUMMARIES
# ============================================================

def render_summaries():

    st.markdown(
        '<div class="section-title">Summaries</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Create concise summaries from your indexed knowledge.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "✨ Generate Knowledge Summary",
        use_container_width=True,
    ):

        with st.spinner("Creating summary..."):

            result = retriever.study_action(
                """
                Create a structured summary of the supplied
                knowledge.

                Organize it into:
                1. Main ideas
                2. Important concepts
                3. Key facts
                4. Useful takeaways

                Use only the supplied context.
                """
            )

        st.markdown("---")
        st.markdown(result)


# ============================================================
# LECTURE MODE
# ============================================================

def render_lecture():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-label">
                LECTURE MODE
            </div>

            <h1>
                Turn lectures into searchable knowledge.
            </h1>

            <p>
                Upload an audio recording through the dashboard
                and let the ingestion pipeline process it.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                🎙️ Supported lecture formats
            </div>

            <div class="card-subtitle">
                MP3 · WAV · M4A · WEBM
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload lecture audio",
        type=["mp3", "wav", "m4a", "webm"],
        accept_multiple_files=True,
    )

    if uploaded:

        if st.button(
            "🎙️ Process Lectures",
            use_container_width=True,
        ):

            progress = st.progress(0)

            for i, file in enumerate(uploaded):

                target = UPLOAD_DIR / file.name
                target.write_bytes(file.getbuffer())

                try:

                    count = ingestion.process_file(target)

                    st.success(
                        f"{file.name} — {count} chunks indexed"
                    )

                except Exception as e:

                    st.error(
                        f"{file.name}: {e}"
                    )

                progress.progress(
                    (i + 1) / len(uploaded)
                )


# ============================================================
# SETTINGS
# ============================================================

def render_settings():

    st.markdown(
        '<div class="section-title">Settings</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Current Voice2Knowledge AI configuration.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="card">

            <div class="card-title">
                🤖 Gemini
            </div>

            <div class="card-subtitle">
                Text model
            </div>

            <p>
                <code>{gemini.text_model}</code>
            </p>

            <div class="card-subtitle">
                Multimodal model
            </div>

            <p>
                <code>{gemini.multimodal_model}</code>
            </p>

            <div class="card-subtitle">
                Embedding model
            </div>

            <p>
                <code>{gemini.embedding_model}</code>
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                🗄️ Knowledge Database
            </div>

            <div class="card-subtitle">
                Chroma vector database
            </div>

            <p>
                <code>data/chroma</code>
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "🗑️ Clear Chat History",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.success("Chat history cleared.")

        st.rerun()


# ============================================================
# ROUTER
# ============================================================

page = st.session_state.page

if page == "Dashboard":

    render_dashboard()

elif page == "Library":

    render_library()

elif page == "Ask":

    render_ask()

elif page == "Study":

    render_study()

elif page == "Summaries":

    render_summaries()

elif page == "Lecture":

    render_lecture()

elif page == "Settings":

    render_settings()

else:

    render_dashboard()