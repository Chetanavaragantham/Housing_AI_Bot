# AUM Housing Community Standards AI Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about Auburn University at Montgomery (AUM) Housing & Residence Life policies. Instead of relying on an LLM's general knowledge, it retrieves the exact relevant sections from the official policy PDF before generating an answer — so responses are always grounded in the actual document.

> 🚀 **[Live Demo →](https://housing-ai-bot-chetana-varagantham.streamlit.app/)**

---

## What it does

- Loads the **AUM-Housing-Community-Standards.pdf** and splits it into searchable chunks
- Embeds chunks using `sentence-transformers/all-mpnet-base-v2` and stores them in **ChromaDB**
- On each question, retrieves the top relevant chunks using **MMR (Maximal Marginal Relevance)**
- Rewrites follow-up questions into standalone queries using conversation history
- Passes retrieved context to **Gemini 2.5 Flash** to generate a grounded answer
- Cites exact **page numbers** and **HRL violation codes** from the document
- Built with a **Streamlit** chat UI with persistent session history

---

## Architecture

```
User question
     │
     ▼
Question rewriter  ←── conversation history
     │
     ▼
ChromaDB (MMR retrieval, k=5, fetch_k=12)
     │
     ▼
Retrieved chunks (with page + HRL code metadata)
     │
     ▼
Gemini 2.5 Flash  ←── grounded answer prompt
     │
     ▼
LLM source selector  ──► cited sources (page, HRL code)
     │
     ▼
Streamlit chat UI
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini 2.5 Flash (`google-genai`) |
| RAG framework | LangChain |
| Document loader | PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `all-mpnet-base-v2` |
| Vector store | ChromaDB (persisted locally) |
| Runtime | Python 3.10+ |

---

## Key Design Decisions

**Why RAG instead of prompting Gemini directly?**
Gemini doesn't know AUM's specific housing policies. RAG grounds every answer in the actual document, prevents hallucination, and makes it easy to cite sources.

**Why MMR retrieval (not plain similarity search)?**
MMR balances relevance with diversity — it avoids returning 5 chunks that all say the same thing, so the LLM gets broader context per query.

**Why chunk_size=500 with chunk_overlap=120?**
Policy documents have short, dense paragraphs. Smaller chunks keep retrieval precise; the overlap ensures no sentence gets cut off at a boundary and loses context.

**Why question rewriting?**
Follow-up questions like "What's the penalty?" are ambiguous without context. Rewriting them into standalone questions before retrieval significantly improves accuracy in multi-turn conversations.

**Why an LLM call for source selection?**
After generating the answer, a second Gemini call identifies which retrieved chunks actually supported the answer. This avoids showing irrelevant citations even when those chunks were technically retrieved.

---

## Setup & Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Chetanavaragantham/Housing_AI_Bot.git
cd Housing_AI_Bot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**

Create a file at `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com/app/apikey).

**4. Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. On first run it builds the ChromaDB index from the PDF (takes ~30 seconds). Subsequent runs load the persisted index instantly.

---

## Project Structure

```
.
├── app.py                              # Main Streamlit app + RAG logic
├── AUM-Housing-Community-Standards.pdf # Source document
├── requirements.txt                    # Python dependencies
├── .streamlit/
│   └── secrets.toml                    # API key (not committed)
└── chroma_db/                          # Persisted vector store (auto-generated)
```

---

## Sample Questions to Try

- *"What happens if a resident is caught with alcohol in the dorms?"*
- *"What is HRL.1002?"*
- *"Can I have a pet in my room?"*
- *"What is the guest policy?"*

---

## About

Built to help AUM housing residents and Resident Assistants (RAs) quickly find answers to policy questions without manually searching through the PDF.
