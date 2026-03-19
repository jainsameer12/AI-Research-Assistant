import streamlit as st
from document_loader import load_documents
from utils.chunking import chunk_documents
from rag_pipeline import build_vector_store, generate_answer

st.title("AI Research Assistant")

# 🔹 Session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload Research Paper", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    documents = load_documents("temp.pdf")

    chunks = chunk_documents(documents)

    # Convert LangChain docs → plain text
    chunks_text = [doc.page_content for doc in chunks]

    build_vector_store(chunks_text)

    st.success("Document processed successfully!")

question = st.text_input("Ask a question")

if question:
    answer, sources = generate_answer(
        question,
        st.session_state.chat_history
    )

    # Store conversation
    st.session_state.chat_history.append((question, answer))

    st.write("### Answer")
    st.write(answer)

    st.write("### Retrieved Context")
    for i, r in enumerate(sources):
        st.write(f"Source {i+1}:")
        st.write(r["text"][:300] + "...")