import streamlit as st
import os
from dotenv import load_dotenv
from rag_app import ingest_pdfs, process_query

load_dotenv()


os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")



# --- Streamlit Page Config ---
st.set_page_config(page_title="AI Writing Assistant", layout="wide")

# --- Sidebar for PDF Upload ---
st.sidebar.header("Upload PDF Files")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    pdf_dir = "user_uploaded_files"
    os.makedirs(pdf_dir, exist_ok=True)

    pdf_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(pdf_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_paths.append(file_path)

    # Process PDFs
    with st.spinner("Processing PDFs..."):
        documents = ingest_pdfs(pdf_paths)
        st.session_state["documents"] = documents  # Store documents in session

    st.sidebar.success("PDFs Processed Successfully!")
# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]

# --- Display Chat Messages ---
st.title("AI Writing Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
query = st.chat_input("Ask a question about your writing...")

if query:
    # Display user message in chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        response = process_query(query)
    if query.strip().lower() == "summary":
        ai_response = response.get("output_text", "I'm sorry, I couldn't generate a summary.")  # Extract only the summary
    # Extract and display only AI-generated response
    ai_response = response.get("generation", "I'm sorry, I couldn't process that.")

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)