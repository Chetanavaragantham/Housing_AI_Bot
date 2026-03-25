import os
import re
import json
import streamlit as st
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Housing Standards AI", page_icon="🏠")
st.title("🏠 Housing Community Standards AI")
st.markdown("Ask questions about AUM Housing & Residence Life community standards.")

# ---------------------------
# Gemini client
# ---------------------------
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

PDF_PATH = "AUM-Housing-Community-Standards.pdf"
PERSIST_DIR = "chroma_db"


# ---------------------------
# Helpers
# ---------------------------
def generate_text(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip() if response.text else ""


def extract_hrl_code(text: str) -> str:
    match = re.search(r"\bHRL\.\d{4}\b", text, re.IGNORECASE)
    return match.group(0).upper() if match else "Not found"


def extract_source_label(doc):
    page_num = doc.metadata.get("page", "N/A")
    hrl_code = extract_hrl_code(doc.page_content)
    return {
        "page": page_num,
        "hr_violation": hrl_code
    }


def format_chat_history(messages, max_turns=6):
    history_lines = []
    recent_messages = messages[-max_turns:]

    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {msg['content']}")

    return "\n".join(history_lines)


def rewrite_question_with_history(question: str, history_text: str) -> str:
    if not history_text.strip():
        return question

    prompt = f"""
You rewrite follow-up questions into standalone questions.

Conversation history:
{history_text}

Latest user question:
{question}

Rewrite the latest user question so it is fully self-contained and clear.
If it is already standalone, return it unchanged.
Return only the rewritten question.
"""
    rewritten = generate_text(prompt)
    return rewritten if rewritten else question


def find_direct_hrl_match(question: str, docs):
    requested = extract_hrl_code(question)
    if requested == "Not found":
        return None

    for doc in docs:
        if extract_hrl_code(doc.page_content) == requested:
            return requested
    return requested


def select_used_sources(answer_text: str, docs):
    sources_text = []
    for i, doc in enumerate(docs, 1):
        label = extract_source_label(doc)
        sources_text.append(
            f"Source {i} | Page {label['page']} | {label['hr_violation']}\n{doc.page_content.strip()}"
        )

    prompt = f"""
You are given an answer and a list of numbered source chunks.

Choose ONLY the source numbers that directly support the final answer.
If the answer says the information was not found, choose the source numbers that best justify that conclusion.

Return JSON only.
Example:
{{"used_sources":[1,3]}}

Answer:
{answer_text}

Sources:
{chr(10).join(sources_text)}
"""
    raw = generate_text(prompt)

    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(json_match.group(0) if json_match else raw)
        used = parsed.get("used_sources", [])
        used = [i for i in used if isinstance(i, int) and 1 <= i <= len(docs)]
        return used if used else [1]
    except Exception:
        return [1]


# ---------------------------
# Load RAG resources
# ---------------------------
@st.cache_resource
def initialize_rag():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    if os.path.exists(PERSIST_DIR):
        db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=PERSIST_DIR
        )

    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 12}
    )


with st.status("Initializing AI engine and PDF context...", expanded=False) as status:
    retriever = initialize_rag()
    status.update(label="Ready! PDF processed and retriever loaded.", state="complete")


# ---------------------------
# Main QA
# ---------------------------
def rag_answer(question: str, messages: list):
    history_text = format_chat_history(messages[:-1], max_turns=6)
    standalone_question = rewrite_question_with_history(question, history_text)

    # Slight retrieval boost for exact HRL lookups
    requested_hrl = extract_hrl_code(standalone_question)
    retrieval_query = standalone_question
    if requested_hrl != "Not found":
        retrieval_query = f"{requested_hrl} {standalone_question}"

    docs = retriever.invoke(retrieval_query)

    numbered_context = []
    for i, doc in enumerate(docs, 1):
        page_num = doc.metadata.get("page", "N/A")
        hrl_code = extract_hrl_code(doc.page_content)
        numbered_context.append(
            f"[Source {i} | Page {page_num} | {hrl_code}]\n{doc.page_content.strip()}"
        )
    context = "\n\n".join(numbered_context)

    answer_prompt = f"""
You are a helpful Housing Community Standards assistant.

Use ONLY the context below to answer the question.
Do not make up policies or details.
If the answer is not clearly supported by the context, say exactly:
"I could not find that in the Housing Community Standards document."

If the user asks about a specific HRL code, answer only if that exact code is supported by the context.

Recent conversation:
{history_text}

Standalone user question:
{standalone_question}

Context:
{context}

Instructions:
- Give a direct answer.
- Keep it clear and concise.
- Do not mention source numbers in the answer text.
- Do not add anything not supported by the context.
"""
    answer = generate_text(answer_prompt)

    used_source_numbers = select_used_sources(answer, docs)
    used_sources = [extract_source_label(docs[i - 1]) for i in used_source_numbers]

    # Deduplicate short citations
    unique_sources = []
    seen = set()
    for src in used_sources:
        key = (src["page"], src["hr_violation"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(src)

    return {
        "answer": answer,
        "standalone_question": standalone_question,
        "sources": unique_sources
    }


# ---------------------------
# Session state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Options")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Responses are grounded in the Housing Community Standards PDF.")


# ---------------------------
# Display chat history
# ---------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View sources"):
                for i, src in enumerate(message["sources"], 1):
                    st.write(f"Source {i} | Page {src['page']} | {src['hr_violation']}")


# ---------------------------
# Chat input
# ---------------------------
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching standards..."):
            result = rag_answer(prompt, st.session_state.messages)

            st.markdown(result["answer"])

            with st.expander("View sources"):
                for i, src in enumerate(result["sources"], 1):
                    st.write(f"Source {i} | Page {src['page']} | {src['hr_violation']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })