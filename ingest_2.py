from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


# Get the folder where ingest.py is located
project_folder = Path(__file__).parent

# Documents folder
documents_folder = project_folder / "documents"

print("Project folder:")
print(project_folder)

print("\nDocuments folder:")
print(documents_folder)

print("\nDocuments folder exists:", documents_folder.exists())


# Find PDF files
pdf_files = list(documents_folder.glob("*.pdf"))

print("\nPDF files found:")

for pdf in pdf_files:
    print("-", pdf.name)


# Store loaded documents
documents = []


# Load each PDF
for pdf_file in pdf_files:

    print(f"\nLoading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))

    pdf_documents = loader.load()

    documents.extend(pdf_documents)

    print("Pages loaded:", len(pdf_documents))


print("\n==============================")
print("Total pages loaded:", len(documents))
print("==============================")


# Check whether documents were loaded
if len(documents) == 0:

    print("\nERROR: No PDF documents were loaded.")
    print("Please check that PDF files are inside:")
    print(documents_folder)

else:

    print("\n========== SAMPLE TEXT ==========")

    print(documents[0].page_content[:2000])

    print("\n========== METADATA ==========")

    print(documents[0].metadata)