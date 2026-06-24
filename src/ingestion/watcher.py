import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.retrieval.vector_store import VectorStore
from src.config import config
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DataFolderHandler(FileSystemEventHandler):
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def on_created(self, event):
        if event.is_directory:
            return
        
        filepath = event.src_path
        filename = os.path.basename(filepath)
        
        # Give the filesystem a moment to finish writing the file
        time.sleep(2)
        
        print(f"👀 Watcher detected new file: {filename}")
        self._ingest_file(filepath, filename)
        
    def on_deleted(self, event):
        if event.is_directory:
            return
            
        filename = os.path.basename(event.src_path)
        print(f"🗑️ Watcher detected deleted file: {filename}")
        self._delete_file_vectors(filename)
        
    def _ingest_file(self, filepath, filename):
        try:
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(filepath)
            else:
                loader = TextLoader(filepath)
                
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
                
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            self.vector_store.index_documents(chunks)
            print(f"✅ Successfully auto-ingested {len(chunks)} chunks for {filename}")
        except Exception as e:
            print(f"❌ Failed to auto-ingest {filename}: {e}")
            
    def _delete_file_vectors(self, filename):
        try:
            from qdrant_client import models
            self.vector_store.client.delete(
                collection_name=self.vector_store.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )
            print(f"✅ Successfully deleted all vectors for {filename}")
        except Exception as e:
            print(f"❌ Failed to delete vectors for {filename}: {e}")

def start_watcher(vector_store: VectorStore):
    data_dir = config.DATA_DIR
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    event_handler = DataFolderHandler(vector_store)
    observer = Observer()
    observer.schedule(event_handler, path=data_dir, recursive=False)
    observer.start()
    print(f"👁️ Started monitoring {data_dir} for changes...")
    return observer
