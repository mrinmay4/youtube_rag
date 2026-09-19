"""
rag_pipeline.py
----------------
All the "brains" of the app live here — no Streamlit/UI code at all.
app.py just calls these functions and displays the results.

This app is a general-purpose RAG system: it can index PDF documents
and/or YouTube video transcripts into ONE shared knowledge base, then
answer questions using only that indexed content.

Pipeline, in two phases:

    PHASE 1 - INDEXING (once per uploaded PDF / YouTube URL)
        source -> extract text -> split into chunks -> embed -> add to FAISS

    PHASE 2 - QUERY (once per question)
        question -> retrieve top-k similar chunks -> LLM -> answer
"""

import os
import re
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi

# ----------------------------------------------------------------------------
# Config constants — easy to point to and explain in an interview.
# ----------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, runs locally/free
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
LLM_MODEL_NAME = "openai/gpt-oss-120b"      # change here if Groq updates model names

# The prompt is the main guardrail against hallucination: answer ONLY from
# the given context, and say so plainly when the answer isn't there.
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


def remove_think_tags(text: str) -> str:
    """Strip any <think>...</think> reasoning blocks some models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ----------------------------------------------------------------------------
# Stage 1a: Load + split PDFs
# ----------------------------------------------------------------------------
def load_and_split_pdfs(uploaded_files):
    """
    Take a list of Streamlit UploadedFile objects, extract text page by
    page, and split into overlapping chunks. Each chunk keeps metadata
    about which file and page it came from, for citing sources later.

    Returns: list of langchain Document objects.
    """
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for uploaded_file in uploaded_files:
        # PyPDFLoader needs a real file path, so write the uploaded bytes
        # to a temporary file first.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()  # one Document per page

            for page in pages:
                page.metadata["source_type"] = "pdf"
                page.metadata["source"] = uploaded_file.name
                page.metadata["page"] = page.metadata.get("page", 0) + 1  # human-friendly, 1-indexed

            all_chunks.extend(splitter.split_documents(pages))
        finally:
            os.remove(tmp_path)

    return all_chunks


# ----------------------------------------------------------------------------
# Stage 1b: Load + split a YouTube video transcript
# ----------------------------------------------------------------------------
def extract_video_id(url: str):
    """Pull the 11-character YouTube video ID out of a URL, or None."""
    match = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def _fetch_transcript_text(video_id: str) -> str:
    """
    Fetch a YouTube video's transcript as one plain-text string.

    youtube-transcript-api changed its interface in v1.0: the old static
    methods (`YouTubeTranscriptApi.get_transcript(...)`) were replaced by
    an instance-based `.fetch(...)`. We try the new interface first and
    fall back to the old one, so this keeps working no matter which
    version ends up installed — that mismatch is what causes "no
    transcript found" errors when LangChain's built-in loader is used
    instead (it only knows the old interface).
    """
    try:
        # youtube-transcript-api >= 1.0
        fetched = YouTubeTranscriptApi().fetch(video_id)
        return " ".join(snippet.text for snippet in fetched)
    except AttributeError:
        # youtube-transcript-api < 1.0
        raw = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(item["text"] for item in raw)


def load_and_split_youtube(url: str):
    """
    Fetch a YouTube video's transcript and split it into overlapping
    chunks. Each chunk is tagged with the video URL as its "source" so
    it can be cited alongside PDF sources.

    Returns: list of langchain Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract a video ID from that URL.")

    transcript_text = _fetch_transcript_text(video_id)
    if not transcript_text.strip():
        raise ValueError("This video's transcript came back empty.")

    doc = Document(
        page_content=transcript_text,
        metadata={"source_type": "youtube", "source": url, "page": "transcript"},
    )
    return splitter.split_documents([doc])


# ----------------------------------------------------------------------------
# Stage 2: Embeddings + FAISS vector store (shared across PDFs + YouTube)
# ----------------------------------------------------------------------------
def get_embedding_model():
    """Sentence-transformer embedding model, loaded once and reused."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def add_chunks_to_store(chunks, existing_store=None):
    """
    Embed the given chunks and either create a new FAISS index or add
    to an existing one. This is what lets PDFs and YouTube transcripts
    live together in a single shared knowledge base — you can keep
    calling this as the user adds more sources.
    """
    embeddings = get_embedding_model()

    if existing_store is None:
        return FAISS.from_documents(chunks, embeddings)

    existing_store.add_documents(chunks)
    return existing_store


# ----------------------------------------------------------------------------
# Stage 3: Retrieval + generation (the actual "RAG" step)
# ----------------------------------------------------------------------------
def get_answer(vector_store, question: str, api_key: str, k: int = 4, temperature: float = 0.0):
    """
    Given the shared vector store and a user question:
      1. Retrieve the top-k most relevant chunks (retrieval)
      2. Stuff them into a prompt along with the question
      3. Ask the LLM to answer strictly from that context (generation)

    Returns a dict: {"answer": str, "sources": list[Document]}
    """
    # --- Retrieval ---
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    relevant_docs = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

    # --- Generation ---
    llm = ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=temperature,
        api_key=api_key,
    )

    chain = ANSWER_PROMPT | llm
    response = chain.invoke({"context": context_text, "question": question})

    return {
        "answer": remove_think_tags(response.content),
        "sources": relevant_docs,
    }


def format_sources(source_docs):
    """
    Turn retrieved Documents into a de-duplicated, readable list of
    source labels for display under the answer, e.g.:
      "report.pdf — page 3"
      "https://youtube.com/watch?v=... — transcript"
    """
    seen = set()
    formatted = []
    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "?")
        label = f"{source} — {page if page == 'transcript' else f'page {page}'}"
        if label not in seen:
            seen.add(label)
            formatted.append(label)
    return formatted
