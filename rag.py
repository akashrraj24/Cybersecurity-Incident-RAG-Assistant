from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


# ==========================================
# STEP 1: Project paths
# ==========================================

project_folder = Path(__file__).parent

chroma_folder = project_folder / "chroma_db"


# ==========================================
# STEP 2: Load embedding model
# ==========================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# STEP 3: Load existing ChromaDB
# ==========================================

print("Loading ChromaDB...")

vectorstore = Chroma(
    persist_directory=str(chroma_folder),
    embedding_function=embeddings
)


# ==========================================
# STEP 4: Create retriever
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# ==========================================
# STEP 5: Load Ollama LLM
# ==========================================

print("Loading Ollama...")

llm = OllamaLLM(
    model="llama3.2"
)


# ==========================================
# STEP 6: Ask question
# ==========================================

question = input("\nAsk a cybersecurity question: ")


# ==========================================
# STEP 7: Retrieve relevant documents
# ==========================================

retrieved_docs = retriever.invoke(question)


# ==========================================
# STEP 8: Combine retrieved text
# ==========================================

context = "\n\n".join(
    document.page_content
    for document in retrieved_docs
)


# ==========================================
# STEP 9: Create RAG prompt
# ==========================================

prompt = f"""
You are a cybersecurity incident response assistant.

Answer the user's question using the provided
cybersecurity documents.

Do not invent information.

If the answer cannot be found in the
provided context, say:

"I could not find this information
in the knowledge base."

Context:
-------------------------
{context}
-------------------------

Question:
{question}

Provide a clear and concise answer.
"""


# ==========================================
# STEP 10: Generate answer
# ==========================================

print("\nGenerating answer...\n")

answer = llm.invoke(prompt)


# ==========================================
# STEP 11: Display answer
# ==========================================

print("==============================")
print("CYBERSECURITY ASSISTANT")
print("==============================")

print(answer)


# ==========================================
# STEP 12: Display sources
# ==========================================

print("\n==============================")
print("SOURCES")
print("==============================")


for document in retrieved_docs:

    source = document.metadata.get("source", "Unknown")
    page = document.metadata.get("page", "Unknown")

    print(f"- {source} | Page: {page}")