"""
app.py
------
Streamlit UI only. All retrieval/LLM logic lives in rag_pipeline.py —
this file wires user actions (add PDF, add YouTube video, ask) to those
functions and displays the results.

Both source types feed into ONE shared FAISS vector store, so a question
can be answered using PDF content, YouTube transcript content, or both.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    load_and_split_pdfs,
    load_and_split_youtube,
    extract_video_id,
    add_chunks_to_store,
    get_answer,
    format_sources,
)


# ============================================================================
# Environment / API key
# ============================================================================

# Load .env from the same directory as this script.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


def get_groq_api_key():
    """
    Get GROQ_API_KEY from Streamlit Cloud Secrets first,
    then fall back to the local .env environment variable.

    Streamlit Cloud:
        st.secrets["GROQ_API_KEY"]

    Local:
        .env -> GROQ_API_KEY=...
    """

    # Try Streamlit Secrets first
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if api_key:
            return api_key, "Streamlit Secrets"
    except Exception:
        # No Streamlit secrets configured
        pass

    # Fallback to environment variable / .env
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key, ".env"

    return None, None


# Get API key
api_key, api_key_source = get_groq_api_key()


# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title="Document & YouTube RAG Assistant",
    page_icon="📚",
    layout="centered",
)


# ============================================================================
# Header
# ============================================================================

st.title("📚 Document & YouTube RAG Assistant")

st.caption(
    "Add PDF documents and/or YouTube videos, then ask questions. "
    "Answers are generated only from what you've added — nothing else."
)


# ============================================================================
# Session state
# ============================================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_sources" not in st.session_state:
    st.session_state.indexed_sources = []

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # API key status
    if api_key:
        st.success(f"Groq API key loaded from {api_key_source}")
    else:
        st.error(
            "GROQ_API_KEY not found. "
            "Add it to Streamlit Secrets or your local .env file."
        )

    # Retrieval parameter
    top_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=2,
        max_value=8,
        value=4,
        help="How many chunks to retrieve from the knowledge base for each question.",
    )

    st.divider()

    # Clear conversation
    if st.session_state.history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.history = []
            st.rerun()

    # Clear all sources
    if st.session_state.indexed_sources:
        if st.button("🗑️ Clear all sources"):
            st.session_state.vector_store = None
            st.session_state.indexed_sources = []
            st.session_state.history = []
            st.rerun()

    st.divider()

    st.caption(
        "Architecture: PDFs / YouTube → chunks → embeddings → "
        "shared FAISS → retrieve → Groq LLM → answer"
    )


# ============================================================================
# Step 1: Add sources
# ============================================================================

st.subheader("1. Add sources")

pdf_tab, youtube_tab = st.tabs(
    ["📄 PDF documents", "🎥 YouTube video"]
)


# ============================================================================
# PDF tab
# ============================================================================

with pdf_tab:

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if st.button(
        "Add PDFs to knowledge base",
        disabled=not uploaded_files,
    ):
        with st.spinner(
            "Extracting text, splitting into chunks, and indexing..."
        ):
            try:
                chunks = load_and_split_pdfs(uploaded_files)

                if not chunks:
                    st.error(
                        "No extractable text was found in the uploaded PDF(s)."
                    )
                else:
                    st.session_state.vector_store = add_chunks_to_store(
                        chunks,
                        st.session_state.vector_store,
                    )

                    st.session_state.indexed_sources.extend(
                        f.name for f in uploaded_files
                    )

                    st.success(
                        f"Indexed {len(chunks)} chunks from "
                        f"{len(uploaded_files)} PDF(s)."
                    )

            except Exception as e:
                st.error(f"Error while processing PDF(s): {e}")


# ============================================================================
# YouTube tab
# ============================================================================

with youtube_tab:

    video_url = st.text_input(
        "YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    video_id = extract_video_id(video_url) if video_url else None

    # Display YouTube thumbnail
    if video_id:
        st.image(
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        )

    elif video_url:
        st.caption(
            "⚠️ That doesn't look like a valid YouTube URL yet."
        )

    # Add YouTube transcript
    if st.button(
        "Add video to knowledge base",
        disabled=not video_id,
    ):
        with st.spinner(
            "Fetching transcript, splitting into chunks, and indexing..."
        ):
            try:
                chunks = load_and_split_youtube(video_url)

                if not chunks:
                    st.error(
                        "No transcript could be found for this video."
                    )
                else:
                    st.session_state.vector_store = add_chunks_to_store(
                        chunks,
                        st.session_state.vector_store,
                    )

                    st.session_state.indexed_sources.append(video_url)

                    st.success(
                        f"Indexed {len(chunks)} chunks "
                        "from the video transcript."
                    )

            except Exception as e:
                st.error(
                    f"Couldn't fetch a transcript for this video: {e}"
                )


# ============================================================================
# Show indexed sources
# ============================================================================

if st.session_state.indexed_sources:

    with st.expander(
        f"📚 Currently indexed "
        f"({len(st.session_state.indexed_sources)})"
    ):
        for source in st.session_state.indexed_sources:
            st.markdown(f"- {source}")


st.divider()


# ============================================================================
# Step 2: Ask questions
# ============================================================================

st.subheader("2. Ask a question")


if st.session_state.vector_store is None:

    st.info(
        "Add at least one PDF or YouTube video above before asking a question."
    )


# ============================================================================
# Display previous conversation
# ============================================================================

for question_text, answer_text, sources in st.session_state.history:

    with st.chat_message("user"):
        st.write(question_text)

    with st.chat_message("assistant"):
        st.write(answer_text)

        if sources:
            with st.expander("📄 Sources used"):
                for source in sources:
                    st.markdown(f"- {source}")


# ============================================================================
# Chat input
# ============================================================================

question = st.chat_input(
    "Ask something about your added documents/videos...",
    disabled=st.session_state.vector_store is None,
)


# ============================================================================
# Process question
# ============================================================================

if question:

    if not api_key:

        st.error(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets or your local .env file."
        )

    else:

        with st.spinner(
            "Retrieving relevant chunks and generating an answer..."
        ):

            try:

                result = get_answer(
                    vector_store=st.session_state.vector_store,
                    question=question,
                    api_key=api_key,
                    k=top_k,
                )

                sources = format_sources(
                    result["sources"]
                )

                st.session_state.history.append(
                    (
                        question,
                        result["answer"],
                        sources,
                    )
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Something went wrong while generating "
                    f"the answer: {e}"
                )