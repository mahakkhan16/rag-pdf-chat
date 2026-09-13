import streamlit as st
import tempfile
import os
from rag_engine import process_pdf, answer_question

st.set_page_config(page_title="Chat With Your PDF", page_icon="📄")
st.title("📄 Chat With Your PDF")
st.write("Upload any PDF and ask questions about its content.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None and st.session_state.vectorstore is None:
    with st.spinner("Processing your document..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.session_state.vectorstore = process_pdf(tmp_path)
        os.unlink(tmp_path)  # clean up temp file

    st.success("Document processed! Ask away.")

# Chat interface
if st.session_state.vectorstore is not None:
    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Question input
    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question(st.session_state.vectorstore, question)
                st.write(result["answer"])

                with st.expander("View source chunks used"):
                    for i, doc in enumerate(result["sources"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})