
# Gen AI RAG Chat Bot using Local Documents

**Gen AI RAG Chat Bot using Local Documents** is a smart chatbot built with **Streamlit** and a **Retrieval-Augmented Generation (RAG)** pipeline.  
It lets you upload PDFs and have a natural conversation with them — you ask questions, and it replies with context-aware, well-grounded answers directly from the document.

---
## Task Preview
![Task Preview](https://image2url.com/images/1761714183830-388ad1fe-d47e-48f6-9f1a-bf3866d4a450.png)


## Features

- **Upload Any PDF:** Drop in any document — research papers, manuals, policies, etc.  
- **Understands Context:** It reads, splits, and organizes text intelligently for better retrieval.  
- **Ask Anything:** You can query naturally, like “What’s the summary of chapter 3?” or “Who is the author mentioning here?”  
- **Real-Time RAG Flow:** Combines retrieval + reranking + LLM generation seamlessly.  
- **Modern UI:** Clean, dark Streamlit interface with animated gradient and chat-style messages.  
- **Live Dashboard:** Sidebar shows insights like number of processed chunks and total queries.  
- **Caching & State Handling:** Uses `st.session_state` and `@st.cache_resource` to optimize performance.  
- **Answer Transparency:** Every response includes number of sources and latency details.

---

## Example Use Cases

- **Research Assistants:** Summarize and analyze multiple academic papers quickly.  
- **Legal Documents:** Extract clauses, cross-reference legal terms, and locate relevant sections efficiently.  
- **Business Reports:** Identify key metrics, financial insights, or performance trends.  
- **Training Manuals:** Retrieve specific definitions, processes, or instructional steps instantly.

---

## How It Works

Here’s the overall flow of how the RAG pipeline runs:

1. **PDF Upload & Text Extraction**  
   - Upload your file via Streamlit.  
   - The app extracts all readable text using `pypdf`.  
   - Each page is tagged with metadata like page number and document name.

2. **Text Chunking**  
   - Instead of dumping all text at once, it’s chunked using a `PremiumPDFChunker`.  
   - Each chunk (≈800 characters) keeps meaning intact, ensuring context accuracy during retrieval.

3. **Embeddings Creation**  
   - Each chunk is converted to numerical vectors using `SentenceTransformer` (`all-MiniLM-L6-v2`).  
   - These embeddings are stored in a local **ChromaDB** vector database for quick semantic search.

4. **Retriever Logic**  
   - The custom retriever ranks chunks using multiple signals:  
     - Keyword similarity  
     - Phrase-level matches  
     - Metadata relevance (like page or section titles)

5. **Reranking**  
   - The top results are reranked using a **CrossEncoder** model (`ms-marco-MiniLM-L-6-v2`).  
   - This ensures the most contextually accurate parts are sent to the LLM.

6. **Answer Generation**  
   - Context chunks are passed into **ChatGroq (LLaMA 3.3 70B)** for final answer generation.  
   - The response is concise, factual, and directly references the source material.

7. **Streamlit Interface**  
   - All interactions happen through a clean chat UI.  
   - Each message (user or assistant) is styled like a real chat bubble for better readability.

---

## Tech Stack Overview

| Layer | Tool / Library |
|-------|----------------|
| **Frontend** | Streamlit |
| **Styling** | Custom CSS (Dark theme with animated gradient) |
| **Vector Store** | ChromaDB |
| **Embeddings** | SentenceTransformer (`all-MiniLM-L6-v2`) |
| **Reranking Model** | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| **LLM** | ChatGroq (`llama-3.3-70b-versatile`) |
| **PDF Processing** | PyPDF |
| **Language** | Python 3.10+ |

---

## Conclusion

- **Gen AI RAG Chat Bot using Local Documents** bridges the gap between static PDFs and interactive knowledge retrieval.  
  It transforms how users interact with documents — replacing manual scrolling and searching with intelligent, conversational exploration.

- This task demonstrates how RAG pipelines combined with LLMs can turn ordinary document reading into an AI-powered discovery experience.  
  In short, it’s not just a chatbot — it’s a personal document analyst built to make complex reading effortless.

---

## Future Work

Here are some areas planned for enhancement:

- **Multi-Document Querying:** Enable the chatbot to process and respond using information from multiple PDF documents simultaneously.  
- **Memory-Based Conversations:** Maintain conversational context across multiple turns for more natural and continuous interactions.  
- **Voice Query Support:** Integrate speech-to-text and text-to-speech functionality for hands-free querying.  
- **Export Chat Summaries:** Allow users to save entire chat sessions as PDF or Markdown reports.  
- **Cloud Deployment:** Deploy the chatbot on cloud platforms such as Hugging Face Spaces or Streamlit Cloud for broader accessibility.
