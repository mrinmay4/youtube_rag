"""
app.py
------

Streamlit UI for a persistent Document + YouTube RAG assistant.

Important change from the previous version:
- st.session_state is still used for the current browser session.
- The FAISS index and indexed-source list are also persisted to disk.
- A fresh browser session restores the previous knowledge base.
- Chat history remains session-specific so users do not see another
  user's conversation.

For Streamlit Community Cloud, the local filesystem is ephemeral. For
true persistence across container restarts/redeployments, set
RAG_PERSIST_DIR to a persistent mounted volume or replace the persistence
functions with object storage/database storage such as S3/Supabase.
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
    load_persistent_state,
    persist_vector_store,
    clear_persistent_state,
)


# ----------------------------------------------------------------------------
# Environment / API key
# ----------------------------------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


def get_groq_api_key():
    """Read GROQ_API_KEY from Streamlit Secrets or local .env."""
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
        if api_key:
            return api_key, "Streamlit Secrets"
    except Exception:
        pass

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key, ".env"

    return None, None


api_key, api_key_source = get_groq_api_key()


# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Document & YouTube RAG Assistant",
    page_icon="📚",
    layout="centered",
)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.title("📚 Document & YouTube RAG Assistant")

st.caption(
    "Add PDF documents and/or YouTube videos, then ask questions. "
    "The knowledge base is persisted so it can be restored in a new session."
)


# ----------------------------------------------------------------------------
# Session + persisted state
# ----------------------------------------------------------------------------

if "_persistence_loaded" not in st.session_state:
    try:
        (
            st.session_state.vector_store,
            st.session_state.indexed_sources,
        ) = load_persistent_state()

    except Exception as e:
        st.session_state.vector_store = None
        st.session_state.indexed_sources = []
        st.warning(
            f"Saved knowledge base could not be loaded: {e}"
        )

    st.session_state.history = []
    st.session_state._persistence_loaded = True

elif "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    if api_key:
        st.success(
            f"Groq API key loaded from {api_key_source}"
        )
    else:
        st.error(
            "GROQ_API_KEY not found. "
            "Add it to Streamlit Secrets or your local .env file."
        )

    top_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=2,
        max_value=8,
        value=4,
        help=(
            "How many chunks to retrieve from the knowledge base "
            "for each question."
        ),
    )

    st.divider()

    if st.session_state.vector_store is not None:
        st.success("Knowledge base: persisted")

    st.caption(
        "Saved index: data/faiss_store/"
    )

    st.divider()

    if st.session_state.history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.history = []
            st.rerun()

    if st.session_state.indexed_sources:
        if st.button("🗑️ Clear all sources"):
            clear_persistent_state()

            st.session_state.vector_store = None
            st.session_state.indexed_sources = []
            st.session_state.history = []

            st.rerun()

    st.divider()

    st.caption(
        "Architecture: PDFs / YouTube → chunks → embeddings → "
        "shared FAISS → retrieve → Groq LLM → answer"
    )

    st.caption(
        "Persistence: FAISS index + source metadata are saved on disk."
    )


# ----------------------------------------------------------------------------
# Step 1: Add sources
# ----------------------------------------------------------------------------

st.subheader("1. Add sources")

pdf_tab, youtube_tab = st.tabs(
    ["📄 PDF documents", "🎥 YouTube video"]
)


# ----------------------------------------------------------------------------
# PDF tab
# ----------------------------------------------------------------------------

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

                    for uploaded_file in uploaded_files:
                        if uploaded_file.name not in st.session_state.indexed_sources:
                            st.session_state.indexed_sources.append(
                                uploaded_file.name
                            )

                    persist_vector_store(
                        st.session_state.vector_store,
                        st.session_state.indexed_sources,
                    )

                    st.success(
                        f"Indexed {len(chunks)} chunks from "
                        f"{len(uploaded_files)} PDF(s) and saved them."
                    )

            except Exception as e:
                st.error(
                    f"Error while processing PDF(s): {e}"
                )


# ----------------------------------------------------------------------------
# YouTube tab
# ----------------------------------------------------------------------------

with youtube_tab:

    video_url = st.text_input(
        "YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    video_id = (
        extract_video_id(video_url)
        if video_url
        else None
    )

    if video_id:
        st.image(
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        )

    elif video_url:
        st.caption(
            "⚠️ That doesn't look like a valid YouTube URL yet."
        )

    if st.button(
        "Add video to knowledge base",
        disabled=not video_id,
    ):
        with st.spinner(
            "Fetching transcript, splitting into chunks, and indexing..."
        ):
            try:
                # Avoid indexing the exact same URL repeatedly.
                if video_url in st.session_state.indexed_sources:
                    st.info(
                        "This video is already in the knowledge base."
                    )
                else:
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

                        st.session_state.indexed_sources.append(
                            video_url
                        )

                        persist_vector_store(
                            st.session_state.vector_store,
                            st.session_state.indexed_sources,
                        )

                        st.success(
                            f"Indexed {len(chunks)} chunks from the "
                            "video transcript and saved them."
                        )

            except Exception as e:
                st.error(
                    f"Couldn't fetch a transcript for this video: {e}"
                )


# ----------------------------------------------------------------------------
# Show indexed sources
# ----------------------------------------------------------------------------

if st.session_state.indexed_sources:

    with st.expander(
        f"📚 Currently indexed "
        f"({len(st.session_state.indexed_sources)})"
    ):
        for source in st.session_state.indexed_sources:
            st.markdown(f"- {source}")

else:
    st.info(
        "No sources are indexed yet. Add a PDF or YouTube video above."
    )


st.divider()


# ----------------------------------------------------------------------------
# Step 2: Ask questions
# ----------------------------------------------------------------------------

st.subheader("2. Ask a question")


# ----------------------------------------------------------------------------
# Display previous conversation
# ----------------------------------------------------------------------------

for question_text, answer_text, sources in st.session_state.history:

    with st.chat_message("user"):
        st.write(question_text)

    with st.chat_message("assistant"):
        st.write(answer_text)

        if sources:
            with st.expander("📄 Sources used"):
                for source in sources:
                    st.markdown(f"- {source}")


# ----------------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------------

question = st.chat_input(
    "Ask something about your added documents/videos...",
    disabled=st.session_state.vector_store is None,
)


# ----------------------------------------------------------------------------
# Process question
# ----------------------------------------------------------------------------

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
