import os
import pickle
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# File to store embeddings
EMBEDDINGS_FILE = "embeddings.pkl"

def create_embeddings(splits):
    """Generate embeddings from document chunks and save to a file."""
    if os.path.exists(EMBEDDINGS_FILE):
        os.remove(EMBEDDINGS_FILE)  # Remove old file if it exists
    
    # Initialize embedding model
    model_name = "BAAI/bge-small-en"
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name)
    
    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Save vectorstore to file
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(vectorstore, f)
    return vectorstore

def load_embeddings():
    """Load previously saved embeddings from file."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError("Embeddings file not found. Please create embeddings first.")
