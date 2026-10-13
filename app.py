import streamlit as st
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Cybersecurity Assistant",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Cybersecurity Incident Response Assistant")
st.write(
    "Ask questions about cybersecurity incident response "
    "using the provided knowledge base."
)


# ==============================
# LOAD PROJECT PATH
# ==============================

project_folder = Path(__file__).parent
chroma_folder = project_folder / "chroma_db"


# ==============================
# LOAD EMBEDDING MODEL
# ==============================

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=str(chroma_folder),
        embedding_function=embeddings
    )

    return vectorstore


# ==============================
# LOAD LLM
# ==============================

@st.cache_resource
def load_llm():

    return OllamaLLM(
        model="llama3.2"
    )


vectorstore = load_vectorstore()
llm = load_llm()


# ==============================
# CHAT HISTORY
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==============================
# USER QUESTION
# ==============================

question = st.chat_input(
    "Ask a cybersecurity question..."
)


if question:

    # Show user question

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # ==============================
    # RETRIEVE DOCUMENTS
    # ==============================

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    retrieved_docs = retriever.invoke(question)


    # ==============================
    # CREATE CONTEXT
    # ==============================

    context = "\n\n".join(
        document.page_content
        for document in retrieved_docs
    )


    # ==============================
    # CREATE PROMPT
    # ==============================

    prompt = f"""
You are a cybersecurity incident response assistant.

Answer the user's question using ONLY
the provided cybersecurity documents.

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


    # ==============================
    # GENERATE ANSWER
    # ==============================

    with st.chat_message("assistant"):

        with st.spinner("Analyzing cybersecurity documents..."):

            answer = llm.invoke(prompt)

        st.markdown(answer)


        # ==============================
        # SHOW SOURCES
        # ==============================

        st.subheader("📚 Retrieved Sources")

        for i, document in enumerate(
            retrieved_docs,
            start=1
        ):

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            st.write(
                f"**Source {i}:** "
                f"{Path(source).name} | "
                f"Page: {page}"
            )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )