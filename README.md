# AUM Housing Community Standards AI Chatbot (RAG)
```yaml
---
title: AUM Housing Community Standards Chatbot
emoji: 🏠
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

A Retrieval-Augmented Generation (RAG) chatbot built for **Auburn University at Montgomery (AUM) Housing & Residence Life**.  
It helps **residents and resident assistants (RAs)** quickly find accurate answers by retrieving relevant sections from the **AUM Housing Community Standards PDF** and generating a short response **based only on the retrieved context**.

## What this chatbot does
- Loads the **AUM-Housing-Community-Standards.pdf**
- Splits the document into searchable chunks
- Creates embeddings using `sentence-transformers/all-mpnet-base-v2`
- Stores embeddings in an in-memory **Chroma** vector database
- Retrieves the **top-k (k=4)** most relevant chunks for each question
- Uses **microsoft/Phi-3-mini-4k-instruct** to generate a concise answer using only retrieved context
- Provides a simple chat UI using **Gradio ChatInterface**

## Tech Stack
- **UI:** Gradio
- **RAG framework:** LangChain
- **Document loader:** PyPDFLoader
- **Chunking:** RecursiveCharacterTextSplitter
- **Embeddings:** HuggingFaceEmbeddings (`all-mpnet-base-v2`)
- **Vector Store:** ChromaDB
- **LLM:** Microsoft Phi-3 Mini 4K Instruct (Transformers)
- **Runtime:** PyTorch

## Repository Structure 

## Deploy to Hugging Face Spaces (Gradio)

This project can be deployed as a **Gradio Space** on Hugging Face. Spaces are Git repos, and configuration is done via a **YAML block at the top of `README.md`**. :contentReference[oaicite:1]{index=1}

### 1) Create a new Space
1. On Hugging Face, create a new **Space**
2. Choose **SDK: Gradio**
3. Set visibility (Public/Private) as needed

When you select Gradio, Spaces will run your `app_file` and install dependencies from `requirements.txt`. :contentReference[oaicite:2]{index=2}


Usage Tips

Ask direct questions about housing rules, policies, violations, guest policy, quiet hours, prohibited items, etc.

The chatbot is designed to answer in one short paragraph.

If the PDF does not contain the answer, results may be incomplete—always verify with official AUM Housing policies for critical decisions.

Limitations

Answers depend on what is present in the PDF and the retrieved chunks.

No conversation memory is used in the current version (history is ignored).

Vector database is built at runtime (not persisted). Restarting the app rebuilds the index.

Future Improvements (optional ideas)

Persist ChromaDB to disk for faster startup

Add citation display (page numbers / chunk sources)

Add conversation memory (optional)

Add evaluation set + retrieval metrics (hit-rate / groundedness)

Add guardrails: refuse if context is missing or ambiguous

Disclaimer

This tool is intended for informational assistance only. For official decisions, refer to AUM Housing & Residence Life documentation or staff.
