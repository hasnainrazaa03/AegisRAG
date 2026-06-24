from typing import List
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from src.config import config

class VectorStore:
    def __init__(self):
        # Using FastEmbed for local, fast embeddings without needing an API key
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.url = config.QDRANT_URL
        self.path = config.QDRANT_PATH
        self.collection_name = config.COLLECTION_NAME
        
        if self.url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif self.url:
            self.client = QdrantClient(url=self.url)
        else:
            # Local persistent storage via path
            self.client = QdrantClient(path=self.path)

        # Ensure collection exists so Langchain doesn't throw a ValueError
        if not self.client.collection_exists(self.collection_name):
            from qdrant_client.models import VectorParams, Distance
            # FastEmbed bge-small-en-v1.5 has size 384
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def index_documents(self, documents: List[Document]) -> QdrantVectorStore:
        """
        Indexes documents into Qdrant.
        """
        qdrant = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        qdrant.add_documents(documents)
        return qdrant
        
    def get_retriever(self, search_kwargs: dict = None):
        """
        Returns a retriever interface for the vector store.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        qdrant = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        return qdrant.as_retriever(search_kwargs=search_kwargs)
