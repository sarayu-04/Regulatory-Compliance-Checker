import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(uploaded_file):
    """Load and process a PDF file."""
    with open("temp_pdf_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return PyPDFLoader("temp_pdf_file.pdf").load()

def split_document(docs):
    """Split a document into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)
