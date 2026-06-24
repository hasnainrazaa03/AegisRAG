from langgraph.graph import StateGraph, END
from src.agents.state import AgenticRAGState
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.critic import CriticAgent

class AgenticRAGWorkflow:
    def __init__(self, llm=None, use_hyde: bool = True, max_iterations: int = 3, vector_store=None):
        self.max_iterations = max_iterations
        self.researcher = ResearcherAgent(llm=llm, use_hyde=use_hyde, vector_store=vector_store)
        self.synthesizer = SynthesizerAgent(llm=llm)
        self.critic = CriticAgent(llm=llm)
        
        self.workflow = StateGraph(AgenticRAGState)
        self._build_graph()

    def _build_graph(self):
        # Add nodes
        self.workflow.add_node("research", self.researcher.invoke)
        self.workflow.add_node("synthesize", self.synthesizer.invoke)
        self.workflow.add_node("critique", self.critic.invoke)
        
        # Define edges
        self.workflow.set_entry_point("research")
        self.workflow.add_edge("research", "synthesize")
        self.workflow.add_edge("synthesize", "critique")
        
        # Conditional edge: If hallucination and iterations < 3, rewrite. Otherwise, end.
        self.workflow.add_conditional_edges(
            "critique",
            self._decide_next_step,
            {
                "rewrite": "synthesize",
                "end": END
            }
        )
        
        self.app = self.workflow.compile()

    def _decide_next_step(self, state: AgenticRAGState):
        if state["is_hallucination"] and state.get("iterations", 0) < self.max_iterations:
            print("--- DECISION: Hallucination detected. Rewriting... ---")
            return "rewrite"
        print("--- DECISION: Answer approved or max iterations reached. ---")
        return "end"
        
    def run(self, question: str):
        print(f"\n[Starting Agentic Workflow] Question: {question}")
        initial_state = {
            "question": question,
            "documents": [],
            "draft_answer": "",
            "critique": "",
            "is_hallucination": False,
            "iterations": 0,
            "final_answer": {}
        }
        
        final_result = self.app.invoke(initial_state)
        return final_result.get("draft_answer", "No answer generated.")

    def stream(self, question: str):
        print(f"\n[Starting Agentic Workflow Stream] Question: {question}")
        initial_state = {
            "question": question,
            "documents": [],
            "draft_answer": "",
            "critique": "",
            "is_hallucination": False,
            "iterations": 0,
            "final_answer": {}
        }
        return self.app.stream(initial_state)
