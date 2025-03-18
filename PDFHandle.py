from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#load a PDF file and split it into text chunks
def process_pdf(file_path):
    try:
        print(f"Processing PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_documents(data)
        for chunk in chunks[:5]:
            print(f"Type: {type(chunk)}, Content: {chunk.page_content[:100]}, Metadata: {chunk.metadata}")
        return chunks
    
    except Exception as e:
        print(f"Error processing PDF {file_path}: {str(e)}")
        return []


