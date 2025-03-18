from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings  
from langchain_core.documents import Document
from typing import List
import logging
import shutil
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
NUM_CTX = 8000  # Increase context size from default 2064 to 8000
# Define the embedding model (Ensure same model is used for storage & retrieval)
EMBEDDING_MODEL = "nomic-embed-text"
PERSIST_DIRECTORY = "./chroma_db"

def clear_chroma_db():
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
            print("Cleared ChromaDB storage.")
        except PermissionError as e:
            print(f"PermissionError: {e}. Close any program using ChromaDB and try again.")
 
# Store Documents in ChromaDB using Ollama Embeddings    
def store_documents(documents: List[Document], collection_name: str = "search_results"):
    #List of LangChain Document objects.
    
    logger.info("Storing documents in ChromaDB using Ollama embeddings...")

    #Clear Previous ChromaDB Data
    clear_chroma_db()
    if not all(isinstance(doc, Document) for doc in documents):
        raise ValueError("store_documents expected List[Document], but got another type.")

    #Log Number of Documents
    print(f"Storing {len(documents)} document objects into ChromaDB...")

    # Use Ollama Embeddings
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Store Full Document Objects in ChromaDB
    try:
        vector_store = Chroma.from_documents(
            documents=documents,               # Pass Document objects
            embedding=embedding_function,      # Use Ollama Embeddings
            collection_name=collection_name,
            persist_directory=PERSIST_DIRECTORY
        )
        print(f"{len(documents)} documents successfully stored in ChromaDB!")
    except Exception as e:
        print(f"Error storing documents in ChromaDB: {str(e)}")
        logger.error(f"Error storing documents: {str(e)}")

#Returns a retriever for the vector store using Ollama embeddings
def get_retriever(collection_name: str = "search_results", k: int = 4):
    """
    Returns a retriever for the vector store using Ollama embeddings.
    
    Args:
        collection_name (str, optional): The name of the Chroma collection. Defaults to "search_results".
        k (int, optional): Number of similar results to return. Defaults to 5.
    
    Returns:
        langchain.vectorstores.base.VectorStoreRetriever: A retriever object.
    """
    logger.info("Creating retriever with Ollama embeddings...")

    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"Documents in ChromaDB: {vector_store._collection.count()}")
    
    #return langchain.vectorstores.base.VectorStoreRetriever which is a retriever object
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
