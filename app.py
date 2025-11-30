import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#loading the pdf from repo

pdf_path = "AUM-Housing-Community-Standards.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

#splitting into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

#embedding the chucks of data using model all-mpnet-base-v2
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
#storing the embeeding vectors in chromaDB
db = Chroma.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

#loadinh the llm model 

model_id = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


def generate_llm_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=350)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def rag_answer(question: str) -> str:
    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    # Build hidden context
    context = "\n\n".join([d.page_content.strip() for d in docs])

    # Hidden prompt
    prompt = f"""
Using ONLY the information in the context, answer the user's question in one clear paragraph.
JUST answer the question directly in one short paragraph.

Context:
{context}

Question: {question}

Answer:
"""

    # Run the model
    raw_output = generate_llm_answer(prompt).strip()

    # Extract ONLY the answer (everything after "Answer:")
    if "Answer:" in raw_output:
        answer = raw_output.split("Answer:")[-1].strip()
    else:
        answer = raw_output.strip()

    return answer


# wrapper

def answer_question(message, history):
    # history is ignored for now, we just use the latest message
    return rag_answer(message)


demo = gr.ChatInterface(
    fn=answer_question,
    title="Housing Community Standards AI ChatBot",
    description="Ask questions about AUM Housing & Residence Life community standards.",
)


if __name__ == "__main__":
    demo.launch()
