# Cybersecurity Incident RAG Assistant

## Project Overview

The Cybersecurity Incident RAG Assistant is a Python-based application that uses Retrieval-Augmented Generation (RAG) to answer questions about cybersecurity incidents using relevant information retrieved from a knowledge base.

## Objectives

- Retrieve relevant information from cybersecurity documents.
- Generate context-aware answers using a Large Language Model (LLM).
- Make cybersecurity incident information easier to search and understand.
- Reduce unsupported answers by grounding responses in retrieved documents.

## Technologies Used

- Python
- LangChain
- Ollama
- Large Language Models (LLMs)
- ChromaDB / Vector Database
- Embeddings
- Retrieval-Augmented Generation (RAG)

*Adjust the technology list to match your actual implementation.*

## How It Works

1. Load cybersecurity incident documents.
2. Split documents into smaller text chunks.
3. Generate embeddings and store them in a vector database.
4. Retrieve relevant chunks when a user asks a question.
5. Pass the retrieved context to the language model.
6. Generate an answer based on the available context.

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Install Ollama and download the language model configured in the application.
5. Run the document ingestion script.
6. Start the application.

## Future Improvements

- Add a web-based user interface.
- Include source references in generated answers.
- Support additional cybersecurity document formats.
- Evaluate retrieval quality and answer accuracy.
