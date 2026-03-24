
# AUM Housing Community Standards AI Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot built for **Auburn University at Montgomery (AUM) Housing & Residence Life**.  
It helps **residents and resident assistants (RAs)** quickly find accurate answers by retrieving relevant sections from the **AUM Housing Community Standards PDF** and generating a short response **based only on the retrieved context**.

## What this chatbot does
- Loads the **AUM-Housing-Community-Standards.pdf**
- Splits the document into searchable chunks
- Creates embeddings using `sentence-transformers/all-mpnet-base-v2`
- Stores embeddings in **ChromaDB**
- Retrieves the **top-k (k=4)** most relevant chunks for each question
- Uses **microsoft/Phi-3-mini-4k-instruct** to generate a concise answer using only retrieved context
- Provides a chat UI using **Gradio ChatInterface**

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
```text
.
├── app.py
├── AUM-Housing-Community-Standards.pdf
├── README.md
└── requirements.txt
