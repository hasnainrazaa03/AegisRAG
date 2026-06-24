from typing import List
from langchain_core.documents import Document
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval.vector_store import VectorStore

class AdvancedRetriever:
    def __init__(self, use_hyde: bool = True, vector_store=None, llm=None):
        self.vector_store = vector_store or VectorStore()
        # Initialize local re-ranker using TinyBERT to avoid ONNX token_type_ids crash in newer onnxruntime
        self.compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
        self.base_retriever = self.vector_store.get_retriever(search_kwargs={"k": 10})
        
        self.use_hyde = use_hyde
        if self.use_hyde:
            # Use provided LLM (Claude/Ollama) or default to local Ollama
            self.llm = llm or ChatOllama(model="llama3")
            self.hyde_prompt = ChatPromptTemplate.from_template(
                "Please write a passage to answer the question. Question: {question}\nPassage:"
            )
            self.hyde_chain = self.hyde_prompt | self.llm | StrOutputParser()

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Executes advanced retrieval using HyDE (optional) and Re-ranking.
        """
        search_query = query
        
        if self.use_hyde:
            try:
                # Generate a hypothetical document to improve dense retrieval recall
                hypothetical_doc = self.hyde_chain.invoke({"question": query})
                search_query = f"{query}\n\n{hypothetical_doc}"
            except Exception as e:
                print(f"HyDE generation failed (is Ollama running?): {e}. Falling back to standard query.")
                
        # 1. Base Retrieval
        base_docs = self.base_retriever.invoke(search_query)
        
        if not base_docs:
            return []
            
        # 2. Re-rank
        # Returns compressed/ranked documents
        reranked_docs = self.compressor.compress_documents(base_docs, search_query)
        return reranked_docs
