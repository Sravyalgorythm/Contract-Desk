import hashlib
from html import escape

import numpy as np
import streamlit as st

from legal_agent.agent import AgentAnswer, LegalAgent
from legal_agent.config import settings
from legal_agent.documents import DocumentError, extract_blocks
from legal_agent.ollama_client import OllamaClient, OllamaError
from legal_agent.retrieval import ContractIndex, chunk_blocks


st.set_page_config(page_title="Contract Desk", page_icon="§", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #202c2a;
        --muted: #66736f;
        --line: #dce4df;
        --surface: #ffffff;
        --canvas: #f5f7f5;
        --sidebar: #edf2ee;
        --green: #246453;
        --green-soft: #e9f2ed;
        --gold: #a77c42;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 245, 0.92);
    }

    .block-container {
        max-width: 1160px;
        padding: 2.25rem 2.5rem 4rem;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebarContent"] {
        padding: 2rem 1.2rem 1.5rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    h1 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.45rem !important;
        font-weight: 600 !important;
        line-height: 1.12 !important;
        margin: 0.25rem 0 0.35rem !important;
    }

    h2 {
        font-size: 1.35rem !important;
        font-weight: 650 !important;
    }

    h3 {
        font-size: 1.05rem !important;
        font-weight: 650 !important;
    }

    p, label, [data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }

    .desk-overline, .field-label, .evidence-kicker {
        color: var(--green);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    .desk-overline {
        color: #154c3e !important;
        font-size: 0.74rem;
        font-weight: 800;
        opacity: 1;
    }

    .desk-header {
        border-bottom: 1px solid var(--line);
        margin: 0 0 1.1rem;
        padding: 0.3rem 0 1.2rem;
    }

    .desk-header p {
        color: var(--muted);
        font-size: 0.98rem;
        margin: 0.35rem 0 0;
    }

    .legal-note {
        align-items: center;
        background: #eff4f0;
        border-left: 3px solid var(--gold);
        border-radius: 0 5px 5px 0;
        color: #46534e;
        display: flex;
        gap: 0.75rem;
        margin: 1rem 0 1.4rem;
        padding: 0.75rem 1rem;
    }

    .legal-note strong {
        color: var(--ink);
        font-size: 0.76rem;
        letter-spacing: 0;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .legal-note span {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .runtime-heading {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0;
        margin: 0 0 0.8rem;
        text-transform: uppercase;
    }

    .model-row {
        border-bottom: 1px solid var(--line);
        margin-bottom: 0.75rem;
        padding: 0 0 0.75rem;
    }

    .model-row span {
        color: var(--muted);
        display: block;
        font-size: 0.72rem;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
    }

    .model-row code {
        background: transparent;
        color: var(--ink);
        font-size: 0.82rem;
        padding: 0;
    }

    .privacy-note {
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.55;
        margin-top: 1.3rem;
        padding-top: 0.9rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface);
        border: 1px dashed #a8b8ae;
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--green);
        background: #fbfdfb;
    }

    [data-testid="stFileUploaderDropzone"] button {
        border: 1px solid var(--line);
        border-radius: 4px;
        color: var(--ink);
    }

    .document-status {
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 6px;
        display: flex;
        gap: 0.9rem;
        justify-content: space-between;
        margin: 0.5rem 0 1.35rem;
        padding: 0.8rem 1rem;
    }

    .document-identity {
        align-items: center;
        display: flex;
        gap: 0.8rem;
        min-width: 0;
    }

    .document-type {
        align-items: center;
        background: var(--green-soft);
        border-radius: 4px;
        color: var(--green);
        display: flex;
        flex: 0 0 2.6rem;
        font-size: 0.7rem;
        font-weight: 800;
        height: 2.6rem;
        justify-content: center;
    }

    .document-name {
        min-width: 0;
    }

    .document-name span {
        color: var(--muted);
        display: block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    .document-name strong {
        color: var(--ink);
        display: block;
        font-size: 0.95rem;
        font-weight: 650;
        overflow-wrap: anywhere;
    }

    .document-meta {
        align-items: center;
        color: var(--muted);
        display: flex;
        flex: 0 0 auto;
        font-size: 0.8rem;
        gap: 0.45rem;
    }

    .document-meta::before {
        background: #6e9a7e;
        border-radius: 50%;
        content: "";
        height: 0.45rem;
        width: 0.45rem;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--line);
        gap: 1.5rem;
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: var(--muted);
        font-size: 0.92rem;
        font-weight: 600;
        height: 3.1rem;
        padding: 0 0.1rem;
    }

    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--green) !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background: var(--green);
        height: 2px;
    }

    [data-testid="stChatMessage"] {
        background: transparent;
        border-bottom: 1px solid var(--line);
        padding: 1rem 0;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        line-height: 1.7;
    }

    [data-testid="stChatInput"] textarea {
        background: var(--surface);
        border: 1px solid #cbd6cf;
        border-radius: 6px;
        min-height: 3.4rem;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--green);
        box-shadow: 0 0 0 1px var(--green);
    }

    [data-testid="stExpander"] {
        background: #f0f4f1;
        border: 1px solid var(--line);
        border-radius: 5px;
        margin-top: 0.75rem;
    }

    .evidence-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 5px;
        margin: 0.6rem 0;
        padding: 0.8rem 0.9rem;
    }

    .evidence-meta {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.5;
        margin-top: 0.35rem;
        overflow-wrap: anywhere;
    }

    button[kind="primary"] {
        background: var(--green);
        border: 1px solid var(--green);
        border-radius: 4px;
        color: #fff;
        font-weight: 650;
    }

    button[kind="primary"]:hover {
        background: #1b5143;
        border-color: #1b5143;
    }

    [data-testid="stAlert"] {
        border-radius: 5px;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1.4rem 1rem 3rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        .document-status {
            align-items: flex-start;
            flex-direction: column;
        }

        .legal-note {
            align-items: flex-start;
            flex-direction: column;
            gap: 0.25rem;
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client() -> OllamaClient:
    return OllamaClient(
        settings.ollama_base_url, settings.chat_model, settings.embedding_model
    )


def build_index(content: bytes, filename: str) -> ContractIndex:
    blocks = extract_blocks(content, filename)
    chunks = chunk_blocks(
        blocks, max_words=settings.chunk_words, overlap_words=settings.overlap_words
    )
    client = get_client()
    vectors = []
    for start in range(0, len(chunks), 32):
        vectors.extend(client.embed([chunk.text for chunk in chunks[start : start + 32]]))
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise OllamaError("Ollama returned an invalid set of document embeddings.")
    return ContractIndex(tuple(chunks), matrix)


def render_evidence(answer: AgentAnswer) -> None:
    if not answer.evidence:
        return
    with st.expander(f"Review evidence · {len(answer.evidence)} passages", expanded=False):
        for number, item in enumerate(answer.evidence, start=1):
            source = "; ".join(
                f"{ref.filename}, {ref.locator}" for ref in item.chunk.sources
            )
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-kicker">Evidence {number:02d} · relevance {item.score:.2f}</div>
                    <div class="evidence-meta">{escape(source)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(item.chunk.text)


def show_answer(answer: AgentAnswer) -> None:
    st.markdown(answer.text)
    render_evidence(answer)


st.markdown(
    """
    <div class="desk-header">
        <div class="desk-overline">Legal operations · Local workspace</div>
        <h1>Contract Desk</h1>
        <p>Private contract review with source passages kept close at hand.</p>
    </div>
    <div class="legal-note">
        <strong>Review support</strong>
        <span>Not legal advice. Verify important conclusions against the original contract.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="runtime-heading">Local runtime</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="model-row"><span>Chat model</span><code>{escape(settings.chat_model)}</code></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="model-row"><span>Embedding model</span><code>{escape(settings.embedding_model)}</code></div>',
        unsafe_allow_html=True,
    )
    relevance = st.slider(
        "Minimum retrieval relevance",
        min_value=0.0,
        max_value=0.9,
        value=min(max(settings.min_relevance, 0.0), 0.9),
        step=0.01,
        help="Raise this to make the agent abstain more often when evidence is weak.",
    )
    st.markdown(
        '<div class="privacy-note">Contract text is processed locally in this session and is not saved by the app.</div>',
        unsafe_allow_html=True,
    )

uploaded_file = st.file_uploader("Upload a contract", type=["pdf", "docx"])
if "document_hash" not in st.session_state:
    st.session_state.document_hash = None
if "contract_index" not in st.session_state:
    st.session_state.contract_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is None:
    st.write("Upload a PDF or DOCX to begin.")
    st.stop()

document_bytes = uploaded_file.getvalue()
document_hash = hashlib.sha256(b"paragraph-chunks-v2" + document_bytes).hexdigest()
if len(document_bytes) > settings.max_upload_bytes:
    st.error("This file is larger than the 30 MB upload limit.")
    st.stop()

if st.session_state.document_hash != document_hash:
    try:
        with st.spinner("Reading the contract and creating local embeddings..."):
            get_client().ensure_models_available()
            contract_index = build_index(document_bytes, uploaded_file.name)
        st.session_state.document_hash = document_hash
        st.session_state.contract_index = contract_index
        st.session_state.contract_name = uploaded_file.name
        st.session_state.chat_history = []
        st.success(f"Indexed {uploaded_file.name}: {len(contract_index.chunks)} passages.")
    except (DocumentError, OllamaError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

index = st.session_state.contract_index
if index is None:
    st.error("The contract could not be indexed. Try uploading it again.")
    st.stop()

file_extension = uploaded_file.name.rsplit(".", 1)[-1].upper()
st.markdown(
    f"""
    <div class="document-status">
        <div class="document-identity">
            <div class="document-type">{escape(file_extension)}</div>
            <div class="document-name">
                <span>Active document</span>
                <strong>{escape(st.session_state.contract_name)}</strong>
            </div>
        </div>
        <div class="document-meta">{len(index.chunks)} passages indexed locally</div>
    </div>
    """,
    unsafe_allow_html=True,
)
agent = LegalAgent(get_client(), index, min_score=relevance)
ask_tab, clauses_tab, review_tab = st.tabs(
    ["Ask the contract", "Clause checklist", "Summary and attention points"]
)

with ask_tab:
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            show_answer(item["answer"])

    question = st.chat_input("Ask about a term, obligation, date, or clause")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the contract..."):
                    answer = agent.ask(question)
                show_answer(answer)
                st.session_state.chat_history.append(
                    {"question": question, "answer": answer}
                )
            except OllamaError as exc:
                st.error(str(exc))

with clauses_tab:
    st.write("Extract payment, termination, confidentiality, liability, and other key terms.")
    if st.button("Extract key clauses", type="primary"):
        try:
            with st.spinner("Reviewing retrieved contract passages..."):
                answer = agent.analyze("clauses")
            show_answer(answer)
        except OllamaError as exc:
            st.error(str(exc))

with review_tab:
    left, right = st.columns(2)
    with left:
        if st.button("Summarize contract", type="primary"):
            try:
                with st.spinner("Preparing a source-grounded summary..."):
                    answer = agent.analyze("summary")
                show_answer(answer)
            except OllamaError as exc:
                st.error(str(exc))
    with right:
        if st.button("Flag attention points"):
            try:
                with st.spinner("Reviewing retrieved terms for attention points..."):
                    answer = agent.analyze("risks")
                show_answer(answer)
            except OllamaError as exc:
                st.error(str(exc))
