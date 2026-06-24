import pytest
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import AdvancedRetriever
from langchain_core.documents import Document
from src.config import config

# Use in-memory Qdrant for tests
config.QDRANT_URL = ":memory:"

def test_vector_store_initialization():
    store = VectorStore()
    assert store.collection_name == "dense_docs"
    assert store.url == ":memory:"

def test_advanced_retriever_initialization():
    store = VectorStore()
    docs = [Document(page_content="Test data", metadata={"source": "test"})]
    store.index_documents(docs)
    
    # Test initialization without HyDE to avoid requiring Ollama in fast CI
    retriever = AdvancedRetriever(use_hyde=False, vector_store=store)
    assert retriever.use_hyde == False
    assert retriever.compressor is not None

def test_dummy_chunk_indexing():
    store = VectorStore()
    docs = [
        Document(page_content="The primary load-bearing beam is made of titanium and can withstand 5000 PSI.", metadata={"source": "manual.pdf"}),
        Document(page_content="Secondary supports use reinforced steel.", metadata={"source": "manual.pdf"})
    ]
    qdrant_instance = store.index_documents(docs)
    assert qdrant_instance is not None
