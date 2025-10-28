# utils/rag_helper.py
import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_pdf_text(pdf_docs):
    """Extract text from a list of uploaded PDF files."""
    try:
        text = ""
        for pdf in pdf_docs:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF text: {e}")
        return ""

def get_text_chunks(text):
    """Split text into manageable chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error splitting text: {e}")
        return []

def get_vector_store(text_chunks, embeddings):
    """Create and return a FAISS vector store from text chunks."""
    try:
        if not embeddings:
            print("Embeddings model not available. Cannot create vector store.")
            return None

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store.as_retriever()
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None