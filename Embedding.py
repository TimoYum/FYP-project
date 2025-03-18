from typing import List
from langchain_ollama import OllamaEmbeddings
import logging
import time


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def embed_documents(texts: List[str], model: str = "nomic-embed-text") -> List[List[float]]:
    print("embed_documents function is being passed through")
    start_time = time.time()

    logger.info("embed_documents function is being passed through")
    # Initialize Ollama embeddings
    embedding_model = OllamaEmbeddings(model=model)

    # Generate embeddings
    embeddings = embedding_model.embed_documents(texts)
    total_time = time.time() - start_time

    print(f"Embedding completed in {total_time:.2f} seconds")
    
    return embeddings

# Embeds user query using the same Ollama embedding model as document embeddings
def embed_query(text: str, model: str = "nomic-embed-text") -> List[float]:

    embedding_function = OllamaEmbeddings(model=model)  # Use Nomic Embedding
    embedding_vector = embedding_function.embed_query(text)  

    return embedding_vector  # Return the embedding as a list of floats


query_embedding = embed_query("What is the summary of the report?")
print(query_embedding)  # Check if you get a valid list of floats
