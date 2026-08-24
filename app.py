import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="YouTube Video Q&A", layout="centered")
st.title("YouTube Transcript RAG Assistant")

# API key comes from .env (GROQ_API_KEY). Fallback to sidebar if not set,
# so the app still works if someone deploys it without a .env file.
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
else:
    st.sidebar.success("Groq API key loaded from .env")

video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
question = st.text_input("Ask a question about the video:", placeholder="What are the main topics discussed?")


def remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@st.cache_resource(show_spinner="Indexing video transcript...")
def build_vector_store(url: str):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


if st.button("Get Answer"):
    if not groq_api_key:
        st.error("Groq API key not found. Add GROQ_API_KEY to your .env file, or enter it in the sidebar.")
    elif not video_url:
        st.error("Please provide a valid YouTube URL.")
    elif not question:
        st.error("Please enter a question.")
    else:
        try:
            with st.spinner("Retrieving information and generating answer..."):
                vector_store = build_vector_store(video_url)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                llm = ChatGroq(
                    model="qwen/qwen3.6-27b",
                    temperature=0.2,
                    api_key=groq_api_key
                )

                prompt = PromptTemplate(
                    template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.
{context}
Question: {question}""",
                    input_variables=['context', 'question']
                )

                def format_docs(docs):
                    return "\n\n".join(d.page_content for d in docs)

                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })

                main_chain = (
                    parallel_chain
                    | prompt
                    | llm
                    | StrOutputParser()
                    | RunnableLambda(remove_think_tags)
                )

                response = main_chain.invoke(question)
                st.markdown("### Answer")
                st.write(response)
        except Exception as e:
            st.error(f"Error processing request: {e}")
