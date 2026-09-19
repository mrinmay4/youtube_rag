# 📚 Document & YouTube RAG Assistant

A general-purpose Retrieval-Augmented Generation (RAG) app: add PDF
documents and/or YouTube videos, ask questions in plain English, and get
answers grounded strictly in that content — with sources cited, and a
clear "not found" response instead of a hallucinated guess.

Works for any domain (not tied to any one subject) — legal docs,
lecture PDFs, tutorial videos, meeting recordings with captions, etc.

## Folder structure

```
document_rag_app/
├── app.py              # Streamlit UI only (add sources, ask, display)
├── rag_pipeline.py      # All RAG logic: load, chunk, embed, store, retrieve, generate
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

`app.py` never touches LangChain internals directly — it only calls
functions from `rag_pipeline.py`. That split means the pipeline could be
reused behind a CLI or an API without touching any UI code.

## Architecture

```
                    INDEXING (runs each time you add a source)
 PDF file    ──▶ extract text per page ──┐
                                          ├──▶ split into chunks ──▶ embed ──▶ add to shared FAISS index
 YouTube URL ──▶ fetch transcript ───────┘

                    QUERY (runs once per question)
 question ──▶ embed question ──▶ similarity search in FAISS ──▶ top-k chunks
                                                                      │
                                                                      ▼
                                    prompt = context + question ──▶ Groq LLM ──▶ answer
```

PDFs and YouTube transcripts are chunked differently but land in the
**same** FAISS index, tagged with `source_type` ("pdf" or "youtube") plus
a `source` (filename or URL) and `page` (page number or "transcript").
That's what lets one question pull relevant context from both a PDF and
a video at the same time, and still cite exactly where each answer came
from.

Key design choices (good interview talking points):

- **Shared vector store, incremental indexing.** `add_chunks_to_store`
  either creates a new FAISS index or appends to the existing one —
  so adding a second PDF or a video doesn't wipe out what's already indexed.
- **FAISS, in-memory, no external DB.** Simple to run and explain; trades
  persistence across restarts for zero infrastructure.
- **`all-MiniLM-L6-v2` sentence-transformer embeddings.** Small, fast,
  runs locally/free — only the LLM call needs an API key.
- **Groq for generation.** Fast inference, generous free tier, simple API.
- **Chunking with overlap (1000 chars, 150 overlap)** keeps related
  sentences together across chunk boundaries.
- **Strict prompt + explicit "not found" instruction** is the main
  anti-hallucination safeguard: the LLM is told to use *only* retrieved
  context and to say plainly when the answer isn't there.
- **`temperature=0.0` by default** for literal, grounded answers (adjustable
  in the sidebar).

## Setup and run locally

```bash
# 1. Unzip and enter the project
cd document_rag_app

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
cp .env.example .env            # Mac/Linux
# copy .env.example .env         # Windows
# then open .env and set GROQ_API_KEY=your_actual_key
# (get a free key at https://console.groq.com/keys)

# 5. Run the app
streamlit run app.py
```

It opens automatically in your browser, usually at `http://localhost:8501`.

**Security note:** never commit `.env` or paste a real API key into
chat/code you share — `.gitignore` already excludes `.env`, but if a key
is ever exposed, rotate it in the Groq console.

## How to use it

1. Under **PDF documents**, upload one or more PDFs and click
   **Add PDFs to knowledge base** — or under **YouTube video**, paste a
   URL and click **Add video to knowledge base**. Repeat for as many
   sources as you like; they all combine into one knowledge base.
2. Type a question in the chat box at the bottom.
3. Read the answer, then expand **Sources used** to see which file/page
   or video it came from.
4. Ask something not covered by any source — it should say the
   information wasn't found rather than making something up.

## Possible extensions (good "what would you add next" answers)

- Swap FAISS for a persistent vector DB (e.g. Chroma with disk storage)
  so the knowledge base survives across app restarts.
- Add per-source toggles ("search only PDFs" / "search only this video").
- Show a similarity/confidence score next to each retrieved chunk.
- Support more source types (web pages, docx, audio via transcription).
