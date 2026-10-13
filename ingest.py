from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


# Folder containing our cybersecurity PDFs
documents_folder = Path("documents")

# Store all loaded documents
documents = []


# Find every PDF in the documents folder
for pdf_file in documents_folder.glob("*.pdf"):

    print(f"\nLoading: {pdf_file.name}")

    # Create PDF loader
    loader = PyPDFLoader(str(pdf_file))

    # Load PDF pages
    pdf_documents = loader.load()

    # Add pages to our main document list
    documents.extend(pdf_documents)

    print(f"Pages loaded: {len(pdf_documents)}")


# Display total number of pages
print("\n==============================")
print("Total pages loaded:", len(documents))
print("==============================")