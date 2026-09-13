import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

def process_pdf(pdf_path):
    """Load a PDF, split it into chunks, and create a searchable vector store."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    return vectorstore

def get_llm():
    """Create the Groq LLM client."""
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def answer_question(vectorstore, question, k=3):
    """Retrieve relevant chunks and ask the LLM to answer using them."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    llm = get_llm()
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": relevant_docs
    }