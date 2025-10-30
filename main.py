import streamlit as st
import os
import tempfile
import re
import time
import random
from typing import List

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_groq import ChatGroq

from pypdf import PdfReader

# Page Config
st.set_page_config(
    page_title="AskMyPDF RAGBOT",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PERFECT CSS - YOUR ORIGINAL EXACT STYLING
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Animated Dark Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 25%, #0f0f0f 50%, #1a1a1a 75%, #0f0f0f 100%) !important;
        background-size: 400% 400% !important;
        animation: gradientShift 20s ease infinite !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    /* KILL ALL BLUE AREAS COMPLETELY */
    .stChatFloatingInputContainer {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 1rem 0 !important;
    }
    
    div[data-testid="stChatInputContainer"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    div[data-testid="stBottom"] {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
    }
    
    .main > div:last-child {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    .stApp > div:last-child {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Remove Avatar Background */
    .stChatMessage > div:first-child {
        background: transparent !important;
    }
    
    /* Clean Title */
    .clean-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0 0.5rem 0;
        color: #ffffff;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 1rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: #1a1a1a !important;
        border-radius: 16px !important;
        border: 1px solid #2a2a2a !important;
        padding: 1.5rem !important;
        margin: 1rem auto !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        max-width: 800px !important;
    }
    
    .stChatMessage:hover {
        background: #1f1f1f !important;
        border-color: #333333 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.5) !important;
    }
    
    .stChatMessage[data-testid="user-message"] {
        border-left: 3px solid #ffffff !important;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        border-left: 3px solid #666666 !important;
    }
    
    .stChatMessage p {
        color: #e8e8e8 !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
        font-weight: 400 !important;
    }
    
    .stChatMessage .stCaption {
        color: #666666 !important;
        font-size: 0.8rem !important;
        margin-top: 0.75rem !important;
    }
    
    /* COMPLETE INPUT BOX FIX - No Blue, No Red */
    .stChatInput {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 25px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        padding: 0.5rem 1rem !important;
        max-width: 700px !important;
        margin: 0 auto !important;
    }
    
    .stChatInput:hover,
    .stChatInput:focus,
    .stChatInput:focus-within,
    .stChatInput:active {
        border: 1px solid #2a2a2a !important;
        border-color: #2a2a2a !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
        outline: none !important;
    }
    
    .stChatInput > div,
    .stChatInput > div > div {
        background: #1a1a1a !important;
        background-color: #1a1a1a !important;
        border: none !important;
    }
    
    .stChatInput input,
    .stChatInput textarea {
        color: #ffffff !important;
        background: #1a1a1a !important;
        background-color: #1a1a1a !important;
        border: none !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 0 !important;
    }
    
    .stChatInput input::placeholder,
    .stChatInput textarea::placeholder {
        color: #666666 !important;
    }
    
    .stChatInput input:hover,
    .stChatInput input:focus,
    .stChatInput input:active,
    .stChatInput textarea:hover,
    .stChatInput textarea:focus,
    .stChatInput textarea:active {
        background: #1a1a1a !important;
        background-color: #1a1a1a !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* Kill all nested element backgrounds and borders */
    .stChatInput *,
    .stChatInput *:hover,
    .stChatInput *:focus,
    .stChatInput *:active {
        background-color: #1a1a1a !important;
        border-color: transparent !important;
        outline: none !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f0f0f !important;
        border-right: 1px solid #2a2a2a !important;
    }
    
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    .stat-box {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    
    .stat-box:hover {
        background: #1f1f1f;
        transform: scale(1.02);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .stat-label {
        color: #888888;
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    
    .stButton button {
        background: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 10px !important;
        padding: 0.7rem 1rem !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background: #1f1f1f !important;
        border-color: #ffffff !important;
        transform: translateX(5px) !important;
    }
    
    .stChatMessage img {
        border-radius: 50% !important;
        background: transparent !important;
    }
    
    .stSpinner > div {
        border-color: #ffffff transparent transparent transparent !important;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f0f0f;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2a2a2a;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #333333;
    }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 8rem;
        max-width: 1200px;
    }
    
    /* ABSOLUTE NUCLEAR - Override Everything */
    * {
        background-image: none !important;
    }
    
    .stChatFloatingInputContainer,
    .stChatFloatingInputContainer > div,
    div[data-testid="stChatFloatingInputContainer"],
    div[data-testid="stBottom"] > div {
        background: #0f0f0f !important;
        background-color: #0f0f0f !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = ""

MODEL_NAME = "llama-3.3-70b-versatile"

# Session State
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'process_trigger' not in st.session_state:
    st.session_state.process_trigger = False
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'doc_title' not in st.session_state:
    st.session_state.doc_title = "Untitled"
if 'chapter_info' not in st.session_state:
    st.session_state.chapter_info = []

# Classes
class PremiumPDFChunker:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:()\-/@]', '', text)
        return text.strip()
    
    def semantic_split(self, text: str) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

class ImprovedRetriever(BaseRetriever):
    documents: List[Document]
    k: int = 6
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scores = []
        
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            exact_match_score = 100 if query_lower in content_lower else 0
            content_words = set(content_lower.split())
            word_overlap = len(query_words.intersection(content_words))
            keyword_count = sum(content_lower.count(word) for word in query_words)
            is_metadata = doc.metadata.get("type") == "metadata"
            metadata_boost = 50 if is_metadata else 0
            total_score = exact_match_score + (word_overlap * 10) + keyword_count + metadata_boost
            scores.append((doc, total_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scores[:self.k]]

def extract_document_info(documents: List[Document]) -> dict:
    info = {"title": "Untitled", "chapters": []}
    if not documents:
        return info
    first_page_text = documents[0].page_content
    lines = first_page_text.strip().split('\n')
    for line in lines[:10]:
        line = line.strip()
        if 5 < len(line) < 120:
            if not any(skip in line.lower() for skip in ['page','author:','by:','date:','copyright']):
                info["title"] = line
                break
    for doc in documents:
        text = doc.page_content
        chapter_matches = re.findall(r'CHAPTER\s+(\d+)[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
        for match in chapter_matches:
            info["chapters"].append({"number": match[0], "name": match[1].strip()[:100], "page": doc.metadata.get("page", "Unknown")})
    return info

def simple_split(documents: List[Document], chunk_size: int = 800) -> List[Document]:
    chunks = []
    for doc in documents:
        text = doc.page_content
        sentences = re.split(r'[.!?]\s+', text)
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
                current_chunk = sentence + ". "
        if current_chunk.strip():
            chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
    return chunks

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("resume_chunks")
    groq_client = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
    return embedder, reranker, client, collection, groq_client

def embed_and_store_chunks(embedder, collection, chunks):
    embeddings = embedder.encode([chunk.page_content for chunk in chunks], show_progress_bar=True)
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk.page_content],
            embeddings=[embeddings[i].tolist()],
            metadatas=[chunk.metadata]
        )

def retrieve_with_reranking(query, embedder, reranker, retriever):
    docs = retriever._get_relevant_documents(query)
    if not docs:
        return []
    candidates = [doc.page_content for doc in docs]
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in ranked_indices[:3]]

def generate_answer(query, embedder, reranker, retriever, groq_client):
    start_time = time.time()
    context_chunks = retrieve_with_reranking(query, embedder, reranker, retriever)
    
    if not context_chunks:
        return {"answer": "I don't have sufficient information to answer that question.", "sources": 0, "latency": time.time()-start_time}
    
    context_text = "\n\n---\n\n".join(context_chunks)
    system_prompt = "You are an elite AI assistant specializing in document analysis. Provide precise, professional responses."
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    
    try:
        result = groq_client.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # CRITICAL FIX: Extract answer properly
        if hasattr(result, 'content'):
            answer = result.content
        elif isinstance(result, dict):
            answer = result.get("content", str(result))
        else:
            answer = str(result)
        
        return {
            "answer": answer,
            "sources": len(context_chunks),
            "latency": round(time.time() - start_time, 2)
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": 0, "latency": time.time() - start_time}

def main():
    # Header
  # Header
    st.markdown('<div class="clean-title">Chat With PDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent document analysis powered by advanced AI</div>', unsafe_allow_html=True)

    # Load models
    embedder, reranker, client, collection, groq_client = load_models()
    
    # Sidebar with Features
    with st.sidebar:
        st.markdown("### 📊 Dashboard")
        
        # Stats
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(st.session_state.chunks) if st.session_state.chunks else 0}</div>
            <div class="stat-label">Knowledge Chunks</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{st.session_state.total_queries}</div>
            <div class="stat-label">Questions Asked</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 💡 Quick Questions")
        samples = [
            "📄 What is this document about?",
            "📝 Summarize key points",
            "🔍 Main topics covered",
            "💼 Important details"
        ]
        
        for q in samples:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.current_question = q.split(' ', 1)[1]
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Actions")
        
        uploaded = st.file_uploader("📤 Upload PDF", type=['pdf'])
        if uploaded:
            if st.button("🚀 Process Document"):
                st.session_state.process_trigger = True
                st.session_state.uploaded_file = uploaded
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("♻️ Clear Cache", use_container_width=True):
            st.session_state.cache = {}
            st.success("Cache cleared!")
    
    # PDF Processing
    if st.session_state.process_trigger and not st.session_state.processing_done:
        st.session_state.process_trigger = False
        with st.spinner("Processing..."):
            try:
                uploaded_file = st.session_state.uploaded_file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
                    f.write(uploaded_file.read())
                    pdf_path = f.name
                
                reader = PdfReader(pdf_path)
                docs = []
                for i, page in enumerate(reader.pages[:50]):
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        docs.append(Document(page_content=text, metadata={"page": i+1}))
                
                if not docs:
                    st.error("No text found")
                    st.stop()
                
                doc_info = extract_document_info(docs)
                st.session_state.doc_title = doc_info["title"]
                st.session_state.chapter_info = doc_info["chapters"]
                
                chunks = simple_split(docs, chunk_size=800)
                if chunks:
                    meta_text = f"Document Title: {st.session_state.doc_title}\n\n"
                    if st.session_state.chapter_info:
                        meta_text += f"Chapters: {len(st.session_state.chapter_info)}\n"
                    chunks.insert(0, Document(page_content=meta_text, metadata={"page": 0, "type": "metadata"}))
                
                st.session_state.chunks = chunks[:150]
                
                try:
                    client.delete_collection("resume_chunks")
                except:
                    pass
                collection = client.create_collection("resume_chunks")
                embed_and_store_chunks(embedder, collection, st.session_state.chunks)
                
                os.unlink(pdf_path)
                st.session_state.retriever = ImprovedRetriever(documents=st.session_state.chunks, k=6)
                st.session_state.processing_done = True
                st.sidebar.success("✅ Ready!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if "metadata" in msg:
                st.caption(f"⚡ {msg['metadata']['latency']}s · {msg['metadata']['sources']} sources")
    
    # Chat Input - Small & Centered
    if prompt := st.chat_input("Ask me anything about the document..."):
        if hasattr(st.session_state, 'current_question'):
            prompt = st.session_state.current_question
            del st.session_state.current_question
        
        if not st.session_state.processing_done:
            st.error("Upload a PDF first")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.total_queries += 1
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                result = generate_answer(prompt, embedder, reranker, st.session_state.retriever, groq_client)
                st.markdown(result["answer"])
                st.caption(f"⚡ {result['latency']}s · {result['sources']} sources")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "metadata": {"sources": result['sources'], "latency": result['latency']}
        })
        
        st.rerun()

if __name__ == "__main__":
    main()