from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def test_environment_variables():
    """Test if required environment variables are set."""
    api_key = os.getenv("PINECONE_API_KEY")
    host = os.getenv("PINECONE_HOST")
    
    if not api_key:
        print("Error: PINECONE_API_KEY is not set in .env file.")
        return False
    if not host:
        print("Error: PINECONE_HOST is not set in .env file.")
        return False
    
    print(f"PINECONE_API_KEY: {api_key[:5]}****** (Partially Hidden for Security)")
    print(f"PINECONE_HOST: {host}")
    return True

if __name__ == "__main__":
    if test_environment_variables():
        print("Environment variables are correctly set!")
    else:
        print("Please check your .env file and try again.")
