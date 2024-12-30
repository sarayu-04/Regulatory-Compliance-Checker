import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Loading environment variables
load_dotenv()

def initialize_pinecone():
    """Set up Pinecone for vector storage."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing in .env file.")

    pc = Pinecone(api_key=api_key)
    index_name = "pdf-index"
    
    # Create or reset index
    if index_name in pc.list_indexes():
        pc.delete_index(index_name)
    pc.create_index(index_name, dimension=384, metric="euclidean")

def upload_embeddings_to_pinecone(vectorstore):
    """Upload embeddings to Pinecone."""
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index = pc.Index("pdf-index")

    vectors = [
        (doc_id, embedding, {"source": "pdf"})
        for doc_id, embedding in zip(vectorstore.docstore._dict.keys(), vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal))
    ]
    index.upsert(vectors=vectors)
