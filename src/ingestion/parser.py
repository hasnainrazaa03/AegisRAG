import os
from typing import List
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

class DocumentParser:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def parse_pdf(self, filename: str) -> List[Document]:
        """
        Parses a PDF using unstructured to maintain semantic elements.
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        elements = partition_pdf(filename=filepath)
        
        # Simple extraction for now: converting elements to LangChain documents
        docs = []
        for el in elements:
            # We can expand this to handle tables and images differently
            text = str(el)
            if len(text.strip()) > 10: # Filter out noise
                docs.append(Document(page_content=text, metadata={"source": filename, "type": type(el).__name__}))
                
        return docs
