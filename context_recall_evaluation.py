import os
import time
import asyncio
import nest_asyncio
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from ragas import RunConfig, evaluate
from ragas.metrics import faithfulness, context_recall
from rag_app import initialize_rag, process_query

# Import RAG System
import rag_app

# Load environment variables
load_dotenv()

# Set OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

# Initialize Ollama Embeddings
eval_embd = OllamaEmbeddings(model="nomic-embed-text")

# Initialize OpenRouter Model for Evaluation
eval_llm = ChatOpenAI(
    model="google/gemini-2.0-pro-exp-02-05:free",
    temperature=0,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
)

# Initialize RAG System
initialize_rag()

# Load Dataset
ds = load_dataset("explodinggradients/ragas-wikiqa", split="train")
df = pd.DataFrame(ds).sample(n=4, random_state=None)  # Randomly select n-number of samples for testing
print(df)

context = df["context"].tolist()
question = df["question"].tolist()

# Store Documents in ChromaDB for Evaluation Purposes
PERSIST_DIRECTORY = "./chroma_db"
vector_store = Chroma(
    collection_name="eval_hugging_face_wikiqa",
    embedding_function=eval_embd,
    persist_directory=PERSIST_DIRECTORY
)

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  
    chunk_overlap=0,
)

docs = []
print("Splitting and storing dataset into ChromaDB for Evaluation...")

# Convert and Store in ChromaDB
for idx, row in tqdm(df.iterrows(), total=len(df)):
    context_text = str(row["context"])  
    text_chunks = text_splitter.split_text(context_text)
    for chunk in text_chunks:
        docs.append(Document(page_content=chunk))

vector_store.add_documents(docs)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})







# Retrieval and Evaluation (Using `rag_app`)
N_EVAL = 4  
QUESTIONS = df["question"][:N_EVAL].tolist()
GROUND_TRUTHS = df["correct_answer"][:N_EVAL].tolist()

evaluation_data = {"question": [], "ground_truth": [], "contexts": [], "response": []}

print("\nRunning RAG-Based Retrieval and Evaluation...")

for i in tqdm(range(N_EVAL)):
    question = QUESTIONS[i]

    # Get response from `rag_app`
    response = process_query(question)

    retrieved_docs = retriever.invoke(question)
    retrieved_texts = [doc.page_content for doc in retrieved_docs] if retrieved_docs else []

    evaluation_data["question"].append(question)
    evaluation_data["ground_truth"].append(GROUND_TRUTHS[i])
    evaluation_data["contexts"].append(retrieved_texts)  
    evaluation_data["response"].append(response)
    
# Convert into RAGAS Dataset
ragas_dataset = Dataset.from_dict(evaluation_data)

# Run RAGAS Evaluation
run_config = RunConfig(max_workers=10, max_wait=180, timeout=120)

print("Evaluating Retrieval Performance...")
context_recall_retrieval_results = evaluate(
    dataset=ragas_dataset,
    metrics=[context_recall],
    run_config=run_config,
    llm=eval_llm,
    embeddings=eval_embd,
)

print("Evaluation Results:", context_recall_retrieval_results)


