import streamlit as st
from src.controllers.pdf_loader import load_pdf, split_document
from src.config.embeddings import create_embeddings, load_embeddings
from src.utils.vectorstore import initialize_pinecone, upload_embeddings_to_pinecone
from src.models.llama_model import query_llama

st.title("AI-Powered PDF Compliance Checker")

# Step 1: File Uploading
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    st.write("Processing the uploaded file...")
    try:
        docs = load_pdf(uploaded_file)
        splits = split_document(docs)
        vectorstore = create_embeddings(splits)
        st.success("PDF processed and embeddings created!")
    except Exception as e:
        st.error(f"Error processing the PDF: {e}")

    # Step 2: Optional Pinecone Upload
    if st.checkbox("Upload embeddings to Pinecone"):
        try:
            initialize_pinecone()
            upload_embeddings_to_pinecone(vectorstore)
            st.success("Embeddings uploaded to Pinecone successfully!")
        except Exception as e:
            st.error(f"Error uploading to Pinecone: {e}")

# Step 3: Query the Model
st.subheader("Ask a Question")
user_query = st.text_input("Enter your question about the uploaded PDF:")
if user_query:
    try:
        vectorstore = load_embeddings()
        response = query_llama(user_query, vectorstore)
        st.subheader("Response from LLaMA")
        st.write(response)
    except Exception as e:
        st.error(f"Error querying the model: {e}")
