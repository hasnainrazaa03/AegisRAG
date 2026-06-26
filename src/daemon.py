import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.config import config
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.retrieval.vector_store import VectorStore

class DataFolderHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.vector_store = VectorStore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def _get_loader(self, file_path):
        if file_path.endswith('.pdf'):
            return PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            return TextLoader(file_path)
        return None

    def on_created(self, event):
        if event.is_directory:
            return
        file_path = event.src_path
        loader = self._get_loader(file_path)
        if loader:
            print(f"🔄 Daemon detected new file: {file_path}. Ingesting...")
            try:
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                self.vector_store.index_documents(chunks)
                print(f"✅ Daemon successfully ingested {len(chunks)} chunks from {file_path}")
            except Exception as e:
                print(f"❌ Daemon failed to ingest {file_path}: {e}")

    def on_deleted(self, event):
        if event.is_directory:
            return
        file_path = event.src_path
        print(f"🗑️ Daemon detected deleted file: {file_path}. Removing from Qdrant...")
        try:
            from qdrant_client.http import models
            self.vector_store.client.delete(
                collection_name=self.vector_store.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=file_path)
                        )
                    ]
                )
            )
            print(f"✅ Daemon successfully deleted points for {file_path}")
        except Exception as e:
            print(f"❌ Daemon failed to delete {file_path} points: {e}")

    def on_modified(self, event):
        if event.is_directory:
            return
        # A modification can be treated as a delete + create
        file_path = event.src_path
        loader = self._get_loader(file_path)
        if loader:
            print(f"🔄 Daemon detected modification in {file_path}. Re-ingesting...")
            self.on_deleted(event)
            self.on_created(event)

def start_daemon():
    print(f"🚀 Starting Auto-Ingestion Daemon on {config.DATA_DIR}")
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)

    event_handler = DataFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, config.DATA_DIR, recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_daemon()
