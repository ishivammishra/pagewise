import streamlit as st
import tempfile
import os
import time
from rag_engine import load_and_split_pdf, create_vector_store, build_qa_chain

# Page configuration
st.set_page_config(page_title="Pagewise", page_icon="📄", layout="centered")

# -----Custom CSS------
st.markdown(
    """
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0f1117;
    color: #e0e0e0;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Title area */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 600;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #6b7280;
    font-size: 0.95rem;
    margin: 0;
}

/* Upload box */
.upload-area {
    background: #1a1d2e;
    border: 1.5px dashed #6366f1;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    margin: 1.2rem 0;
    transition: border-color 0.2s;
}

/* Status badge */
.status-badge {
    display: inline-block;
    background: #1e2a1e;
    color: #4ade80;
    border: 1px solid #4ade80;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* Chat messages */
.stChatMessage {
    background: #1a1d2e !important;
    border-radius: 12px !important;
    border: 1px solid #2d2f45 !important;
    margin-bottom: 0.6rem !important;
    padding: 0.8rem !important;
}

/* User message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid #6366f1 !important;
}

/* Assistant message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #8b5cf6 !important;
}

/* Chat input */
.stChatInput textarea {
    background: #1a1d2e !important;
    border: 1.5px solid #6366f1 !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stChatInput textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13151f !important;
    border-right: 1px solid #2d2f45 !important;
}
[data-testid="stSidebar"] * {
    color: #9ca3af !important;
}

/* Sidebar header */
.sidebar-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6366f1 !important;
    margin-bottom: 0.8rem;
}

/* Step items in sidebar */
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    color: #9ca3af;
}
.step-number {
    background: #6366f1;
    color: white !important;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 600;
    flex-shrink: 0;
    margin-top: 1px;
}

/* Reset button */
.stButton button {
    background: transparent !important;
    border: 1px solid #374151 !important;
    color: #6b7280 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 0.3rem 0.8rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    border-color: #6366f1 !important;
    color: #a78bfa !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1d2e !important;
    border: 1.5px dashed #6366f1 !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #6366f1 !important;
}

/* Divider */
hr {
    border-color: #2d2f45 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
</style>
""",
    unsafe_allow_html=True,
)

# ---Hero Header----
st.markdown(
    """
<div class="hero">
    <h1>📄 Pagewise</h1>
    <p>Ask anything about your document — powered by LLaMA 3.1 + RAG</p>
</div>
""",
    unsafe_allow_html=True,
)

# ------Sidebar---------
with st.sidebar:
    st.markdown('<div class="sidebar-title">How it works</div>', unsafe_allow_html=True)
    steps = [
        "Upload any PDF",
        "Split into chunks",
        "Stored as vectors",
        "Question finds chunks",
        "LLM answers",
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(
            f"""
        <div class="step-item">
            <div class="step-number">{i}</div>
            <span>{step}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="sidebar-title">Stack</div>', unsafe_allow_html=True)
    st.markdown(
        """
    <div style="font-size:0.82rem; line-height:1.8; color:#6b7280;">
        🔗 LangChain<br>
        ⚡ Groq LLaMA 3.1<br>
        🧠 HuggingFace Embeddings<br>
        🗄️ ChromaDB<br>
        🎈 Streamlit
    </div>
    """,
        unsafe_allow_html=True,
    )

# -----------Session State--------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ------------File Upload------------------
uploaded_file = st.file_uploader(
    "Upload your PDF", type="pdf", label_visibility="collapsed"
)

if uploaded_file:

    if st.session_state.pdf_name != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                chunks = load_and_split_pdf(tmp_path)

                if not chunks:
                    st.error("PDF contains no readable text.")
                    st.stop()

                vector_store = create_vector_store(chunks)
                st.session_state.qa_chain = build_qa_chain(vector_store)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []
                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                st.stop()

    st.markdown(
        f'<div class="status-badge">✓ {uploaded_file.name}</div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask anything about your PDF...")

    if question:

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            with st.spinner(""):
                try:
                    result = st.session_state.qa_chain.invoke({"query": question})
                    full_answer = result["result"]
                except Exception as e:
                    full_answer = f"Error: {e}"

            # Typing animation
            displayed = ""
            for char in full_answer:
                displayed += char
                message_placeholder.markdown(displayed + "▌")
                time.sleep(0.008)  # speed of typing — lower = faster

            message_placeholder.markdown(full_answer)

        st.session_state.messages.append({"role": "assistant", "content": full_answer})

    # Reset button
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if st.button("↺ Reset"):
            st.session_state.qa_chain = None
            st.session_state.messages = []
            st.session_state.pdf_name = None
            st.rerun()

# No file uploaded state
else:
    st.markdown(
        """
    <div style="text-align:center; padding: 3rem 0; color: #374151;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📂</div>
        <div style="font-size: 1rem; color: #6b7280;">Upload a PDF above to start chatting</div>
        <div style="font-size: 0.82rem; color: #374151; margin-top: 0.5rem;">
            Try your resume, a research paper, or any document
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
