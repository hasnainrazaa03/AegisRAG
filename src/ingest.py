import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.retrieval.vector_store import VectorStore
from src.config import config

def ingest_documents():
    print(f"Scanning directory: {config.DATA_DIR}")
    
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
        print(f"Created empty directory {config.DATA_DIR}. Please add PDF or TXT files and run again.")
        return

    documents = []
    
    # Load PDFs
    pdf_files = glob.glob(os.path.join(config.DATA_DIR, "**/*.pdf"), recursive=True)
    for file_path in pdf_files:
        print(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    # Load Text files
    txt_files = glob.glob(os.path.join(config.DATA_DIR, "**/*.txt"), recursive=True)
    for file_path in txt_files:
        print(f"Loading TXT: {file_path}")
        loader = TextLoader(file_path)
        documents.extend(loader.load())

    if not documents:
        print("No documents found in the data directory.")
        return

    print(f"Loaded {len(documents)} raw pages/files.")
    print("Splitting documents into chunks...")
    
    # Dense engineering docs need overlapping chunks to retain context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} overlapping chunks.")
    
    print("Initializing Qdrant Vector Store and creating embeddings...")
    vector_store = VectorStore()
    
    print("Indexing chunks into Qdrant...")
    vector_store.index_documents(chunks)
    
    print("Ingestion complete! The agents are now ready to search your documents.")

if __name__ == "__main__":
    ingest_documents()
