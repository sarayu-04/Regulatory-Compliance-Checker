from langchain_community.llms import Ollama

def query_llama(query, vectorstore):
    """Query LLaMA model using relevant context."""
    llm = Ollama(model="llama3.1")
    
    # Extract context from the vectorstore
    results = vectorstore.similarity_search(query, k=5)
    context = "\n".join([res.page_content for res in results])
    
    # Construct the prompt
    prompt = f"Context:\n{context}\n\nUser's question:\n{query}"
    return llm(prompt)
