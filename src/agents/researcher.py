from src.agents.state import AegisRAGState
from src.retrieval.hybrid_search import AdvancedRetriever
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document

class ResearcherAgent:
    def __init__(self, llm=None, use_hyde: bool = True, vector_store=None):
        self.retriever = AdvancedRetriever(use_hyde=use_hyde, llm=llm, vector_store=vector_store)
        self.search_tool = DuckDuckGoSearchRun()

    def invoke(self, state: AegisRAGState) -> dict:
        """
        Takes the current state, retrieves relevant documents using the advanced hybrid search.
        If no documents are found, it falls back to a DuckDuckGo web search.
        """
        print(f"--- RESEARCHER: Retrieving context for '{state['question']}' ---")
        docs = self.retriever.get_relevant_documents(state["question"])
        
        if not docs:
            print("--- RESEARCHER: No local context found. Falling back to Web Search ---")
            try:
                search_result = self.search_tool.invoke(state["question"])
                docs = [Document(page_content=search_result, metadata={"source": "DuckDuckGo Web Search"})]
            except Exception as e:
                print(f"Web search failed: {e}")
        
        return {"documents": docs}
