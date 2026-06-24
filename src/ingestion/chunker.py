from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Fallback to recursive character splitting if true semantic embeddings aren't available yet
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunks a list of documents into smaller pieces while maintaining metadata.
        """
        return self.splitter.split_documents(documents)
