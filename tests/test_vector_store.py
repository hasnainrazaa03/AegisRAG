import pytest
from src.retrieval.vector_store import VectorStore
from langchain_core.documents import Document

def test_vector_store_initialization():
    # Because config has been patched, this should initialize
    store = VectorStore()
    assert store.collection_name == "dense_docs"
    assert store.client is not None

def test_vector_store_indexing():
    store = VectorStore()
    docs = [
        Document(page_content="Test data 1", metadata={"source": "test1"}),
        Document(page_content="Test data 2", metadata={"source": "test2"})
    ]
    
    # Should not throw
    qdrant = store.index_documents(docs)
    assert qdrant is not None
    
    # Check that retrieving works
    retriever = store.get_retriever(search_kwargs={"k": 2})
    results = retriever.invoke("Test data")
    assert len(results) > 0
    assert "Test data" in results[0].page_content
