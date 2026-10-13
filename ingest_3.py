from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==========================================
# STEP 1: Project and documents folder
# ==========================================

project_folder = Path(__file__).parent

documents_folder = project_folder / "documents"

chroma_folder = project_folder / "chroma_db"


# ==========================================
# STEP 2: Find PDF files
# ==========================================

pdf_files = list(documents_folder.glob("*.pdf"))

print("PDF files found:")

for pdf in pdf_files:
    print("-", pdf.name)


# ==========================================
# STEP 3: Load PDFs
# ==========================================

documents = []

for pdf_file in pdf_files:

    print(f"\nLoading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))

    pdf_documents = loader.load()

    documents.extend(pdf_documents)

    print("Pages loaded:", len(pdf_documents))


print("\nTotal pages:", len(documents))


# ==========================================
# STEP 4: Split documents into chunks
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("\nTotal chunks:", len(chunks))


# ==========================================
# STEP 5: Create embedding model
# ==========================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ==========================================
# STEP 6: Store embeddings in ChromaDB
# ==========================================

print("\nCreating ChromaDB...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(chroma_folder)
)

print("ChromaDB created successfully!")


# ==========================================
# STEP 7: Test similarity search
# ==========================================

query = "What should I do during a ransomware incident?"

results = vectorstore.similarity_search(
    query,
    k=3
)


print("\n================================")
print("SEARCH RESULTS")
print("================================")


for i, result in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")

    print(result.page_content[:500])

    print("\nSource:")
    print(result.metadata)