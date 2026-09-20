"""
rag_pipeline.py
---------------

Core RAG logic plus persistence.

Pipeline:
    source -> extract text -> chunks -> embeddings -> FAISS
    question -> retrieve top-k -> Groq LLM -> answer

Persistence:
    The FAISS index and indexed-source metadata are saved to disk.
    This means the knowledge base can be restored after a Streamlit
    rerun or a new browser session (as long as the deployment filesystem
    itself is persistent).
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# Change this path with RAG_PERSIST_DIR when deploying to a machine/volume
# where you want the index to survive application restarts.
PERSIST_DIR = Path(os.getenv("RAG_PERSIST_DIR", "data"))
FAISS_DIR = PERSIST_DIR / "faiss_store"
STATE_FILE = PERSIST_DIR / "state.json"

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about a set of documents
(these may be PDFs, YouTube video transcripts, or both).

Use ONLY the context below to answer the question. Do not use any outside
knowledge, and do not guess. If the answer cannot be found in the context,
respond exactly with:

"I could not find this information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""
)


# ----------------------------------------------------------------------------
# Small persistence helpers
# ----------------------------------------------------------------------------

def _write_json_atomic(path: Path, data) -> None:
    """Write JSON atomically so a partial write is less likely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _read_json(path: Path, default):
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_persistent_state():
    """
    Restore the saved FAISS index and indexed-source list.

    Returns:
        (vector_store_or_none, indexed_sources)
    """
    indexed_sources = _read_json(
        STATE_FILE,
        {"indexed_sources": []},
    ).get("indexed_sources", [])

    index_file = FAISS_DIR / "index.faiss"
    docstore_file = FAISS_DIR / "index.pkl"

    if not index_file.exists() or not docstore_file.exists():
        return None, indexed_sources

    embeddings = get_embedding_model()

    # allow_dangerous_deserialization is required by LangChain's FAISS loader
    # because the docstore is stored as a pickle created by save_local().
    # Only load files created/controlled by this application.
    vector_store = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store, indexed_sources


def persist_vector_store(vector_store, indexed_sources) -> None:
    """
    Persist the complete FAISS index plus the list of indexed sources.
    """
    if vector_store is None:
        return

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_DIR))

    _write_json_atomic(
        STATE_FILE,
        {
            "version": 1,
            "indexed_sources": indexed_sources,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def clear_persistent_state() -> None:
    """Delete the persisted FAISS index and metadata."""
    if FAISS_DIR.exists():
        shutil.rmtree(FAISS_DIR)

    if STATE_FILE.exists():
        STATE_FILE.unlink()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def remove_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ----------------------------------------------------------------------------
# Stage 1a: Load + split PDFs
# ----------------------------------------------------------------------------

def load_and_split_pdfs(uploaded_files):
    """
    Extract text page-by-page and split it into overlapping chunks.
    Each chunk keeps source and page metadata.
    """
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            for page in pages:
                page.metadata["source_type"] = "pdf"
                page.metadata["source"] = uploaded_file.name
                page.metadata["page"] = page.metadata.get("page", 0) + 1

            all_chunks.extend(splitter.split_documents(pages))
        finally:
            os.remove(tmp_path)

    return all_chunks


# ----------------------------------------------------------------------------
# Stage 1b: Load + split YouTube transcript
# ----------------------------------------------------------------------------

def extract_video_id(url: str):
    """Pull the 11-character YouTube video ID from a URL."""
    match = re.search(
        r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})",
        url,
    )
    return match.group(1) if match else None


PREFERRED_LANGUAGES = ["en", "en-US", "en-GB"]


def _extract_text(fetched) -> str:
    """
    Convert fetched transcript objects/dicts into plain text.
    """
    try:
        return " ".join(snippet.text for snippet in fetched)
    except AttributeError:
        return " ".join(item["text"] for item in fetched)


def _fetch_transcript_text(video_id: str) -> str:
    """
    Fetch a YouTube transcript with compatibility for old/new
    youtube-transcript-api interfaces.
    """
    try:
        api = YouTubeTranscriptApi()

        try:
            fetched = api.fetch(
                video_id,
                languages=PREFERRED_LANGUAGES,
            )
        except Exception:
            transcript_list = api.list(video_id)
            fetched = next(iter(transcript_list)).fetch()

    except AttributeError:
        try:
            fetched = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=PREFERRED_LANGUAGES,
            )
        except Exception:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            fetched = next(iter(transcript_list)).fetch()

    return _extract_text(fetched)


def load_and_split_youtube(url: str):
    """
    Fetch a video's transcript and split it into overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    video_id = extract_video_id(url)

    if not video_id:
        raise ValueError("Could not extract a video ID from that URL.")

    transcript_text = _fetch_transcript_text(video_id)

    if not transcript_text.strip():
        raise ValueError("This video's transcript came back empty.")

    doc = Document(
        page_content=transcript_text,
        metadata={
            "source_type": "youtube",
            "source": url,
            "page": "transcript",
        },
    )

    return splitter.split_documents([doc])


# ----------------------------------------------------------------------------
# Stage 2: Embeddings + FAISS
# ----------------------------------------------------------------------------

try:
    import streamlit as st
except ImportError:
    st = None


def get_embedding_model():
    """
    Load the embedding model once when Streamlit is available.
    Falls back to a normal cached singleton outside Streamlit.
    """
    if st is not None:
        return _get_embedding_model_streamlit()

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


if st is not None:

    @st.cache_resource(show_spinner=False)
    def _get_embedding_model_streamlit():
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def add_chunks_to_store(chunks, existing_store=None):
    """
    Embed new chunks and either create or extend the shared FAISS index.
    """
    embeddings = get_embedding_model()

    if existing_store is None:
        return FAISS.from_documents(chunks, embeddings)

    existing_store.add_documents(chunks)
    return existing_store


# ----------------------------------------------------------------------------
# Stage 3: Retrieval + generation
# ----------------------------------------------------------------------------

def get_answer(
    vector_store,
    question: str,
    api_key: str,
    k: int = 4,
    temperature: float = 0.0,
):
    """
    Retrieve top-k chunks and ask the Groq LLM to answer only from context.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    relevant_docs = retriever.invoke(question)

    context_text = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=temperature,
        api_key=api_key,
    )

    chain = ANSWER_PROMPT | llm

    response = chain.invoke(
        {
            "context": context_text,
            "question": question,
        }
    )

    return {
        "answer": remove_think_tags(response.content),
        "sources": relevant_docs,
    }


# ----------------------------------------------------------------------------
# Source formatting
# ----------------------------------------------------------------------------

def format_sources(source_docs):
    """
    Turn retrieved Documents into a de-duplicated, readable list.
    """
    seen = set()
    formatted = []

    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "?")

        label = (
            f"{source} — "
            f"{page if page == 'transcript' else f'page {page}'}"
        )

        if label not in seen:
            seen.add(label)
            formatted.append(label)

    return formatted
