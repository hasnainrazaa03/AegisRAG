from src.agents.state import AgenticRAGState
from src.retrieval.hybrid_search import AdvancedRetriever

class ResearcherAgent:
    def __init__(self, llm=None, use_hyde: bool = True, vector_store=None):
        self.retriever = AdvancedRetriever(use_hyde=use_hyde, llm=llm, vector_store=vector_store)

    def invoke(self, state: AgenticRAGState) -> dict:
        """
        Takes the current state, retrieves relevant documents using the advanced hybrid search,
        and updates the state.
        """
        print(f"--- RESEARCHER: Retrieving context for '{state['question']}' ---")
        docs = self.retriever.get_relevant_documents(state["question"])
        
        return {"documents": docs}
