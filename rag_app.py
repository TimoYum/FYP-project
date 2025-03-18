import logging
import os
import time

import Agents
import VectorStore_Retrieve
import Flow
from PDFHandle import process_pdf  # Function to process PDFs
from VectorStore_Retrieve import store_documents  # For storing PDFs in ChromaDB
from Flow import Workflow  # For initializing the RAG workflow
from langchain_core.documents import Document

import streamlit as st

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global variables for storing state
workflow = True

# Global variable to track if PDFs were processed
pdfs_processed = False

#Initialize the RAG system
def initialize_rag(model: str = "meta-llama/llama-3.1-8b-instruct:free"):
    
    logger.info("Initializing RAG system...")

    # Load AI Agents
    (
            retrieval_grader,
            summary_chain,
            rag_chain,
            hallucination_grader,
            answer_grader,
            web_search_tool,
            writing_feedback_agent
    ) = Agents.get_agents(model) #model = LLM model to use
        



def ingest_pdfs(pdf_file_paths: list):
    global workflow, pdfs_processed

    # Return Early if Already Processed
    if pdfs_processed:
        if "documents" in st.session_state and isinstance(st.session_state["documents"], list):
            print(f"Documents already exist in session_state with {len(st.session_state['documents'])} documents.")
        else:
            print("PDFs already processed but session_state has no valid documents.")
        return st.session_state.get("documents", [])

    # Process PDFs
    all_chunks = []
    for file_path in pdf_file_paths:
        chunks = process_pdf(file_path)  # List[Document]
        all_chunks.extend(chunks)

    print(f"Extracted {len(all_chunks)} chunks from PDFs")

    # Store Proper Document Objects
    store_documents(documents=all_chunks)  # Store in ChromaDB

    # Save Document Objects in Session State
    if isinstance(all_chunks, list) and all(isinstance(doc, Document) for doc in all_chunks):
        st.session_state["documents"] = all_chunks  # Properly store Documents
        print(f" Stored {len(all_chunks)} documents in session_state. Type: {type(st.session_state['documents'])}")
    else:
        print(f" Invalid document format: {type(all_chunks)}")

    #  Build Workflow
    workflow = Flow.Workflow("meta-llama/llama-3.1-8b-instruct:free").build()
    pdfs_processed = True

    print(" RAG workflow initialized!")
    return all_chunks  #  Return the document objects

def process_query(query: str, retries: int = 2):
    global workflow

    if not workflow:
        return " No documents loaded. Please ingest PDFs first."

    logger.info(f"Processing query: {query}")
    if "documents" not in st.session_state or not isinstance(st.session_state["documents"], list):
        return " No documents found. Please upload PDFs first."

    # Add print logs before workflow.invoke
    print(f" session_state['documents'] Type: {type(st.session_state['documents'])}")
    print(f" session_state['documents'] Length: {len(st.session_state['documents'])}")
    print(f" First Document Example: {st.session_state['documents'][0]}")

    #  Pass session_state["documents"] directly
    response = workflow.invoke({
        "question": query,
        "documents": st.session_state["documents"],  #  Correct
        "retries": retries
    })

    print(f" Workflow Response: {response}")
    # response = workflow.invoke({"question": query, "retries": retries})
    return response #  Return the response as string

def clear_system():
    """
    Clears the vector store and resets the workflow.
    """
    global workflow
    workflow = None
    logger.info(" Cleared vector store and workflow instance.")


