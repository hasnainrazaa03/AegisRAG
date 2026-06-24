from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.agents.state import AegisRAGState
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.supervisor import SupervisorAgent
from src.agents.critic import CriticAgent
from src.agents.grader import GraderAgent

class AegisRAGWorkflow:
    def __init__(self, llm=None, use_hyde: bool = True, max_iterations: int = 3, vector_store=None):
        self.max_iterations = max_iterations
        self.supervisor = SupervisorAgent(llm=llm)
        self.researcher = ResearcherAgent(llm=llm, use_hyde=use_hyde, vector_store=vector_store)
        self.grader = GraderAgent(llm=llm)
        self.synthesizer = SynthesizerAgent(llm=llm)
        self.critic = CriticAgent(llm=llm)
        
        self.workflow = StateGraph(AegisRAGState)
        self._build_graph()

    def _build_graph(self):
        # Add nodes
        self.workflow.add_node("supervisor", self.supervisor.invoke)
        self.workflow.add_node("research", self.researcher.invoke)
        self.workflow.add_node("grade", self.grader.invoke)
        self.workflow.add_node("synthesize", self.synthesizer.invoke)
        self.workflow.add_node("critique", self.critic.invoke)
        
        # Define edges
        self.workflow.set_entry_point("supervisor")
        
        self.workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "research": "research",
                "end": END
            }
        )
        
        self.workflow.add_edge("research", "grade")
        self.workflow.add_edge("grade", "synthesize")
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
        
        self.memory = MemorySaver()
        self.app = self.workflow.compile(
            checkpointer=self.memory,
            interrupt_before=["synthesize"]
        )

    def _route_from_supervisor(self, state: AegisRAGState):
        if state.get("needs_rag", True):
            return "research"
        return "end"

    def _decide_next_step(self, state: AegisRAGState):
        if state["is_hallucination"] and state.get("iterations", 0) < self.max_iterations:
            print("--- DECISION: Hallucination detected. Rewriting... ---")
            return "rewrite"
        print("--- DECISION: Answer approved or max iterations reached. ---")
        return "end"
        
    def run(self, question: str, chat_history: list = None):
        if chat_history is None:
            chat_history = []
        print(f"\n[Starting AegisRAG Workflow] Question: {question}")
        initial_state = {
            "question": question,
            "chat_history": chat_history,
            "documents": [],
            "draft_answer": "",
            "critique": "",
            "is_hallucination": False,
            "needs_rag": True,
            "iterations": 0,
            "final_answer": {}
        }
        
        final_result = self.app.invoke(initial_state)
        return final_result.get("draft_answer", "No answer generated.")

    def stream(self, question: str, chat_history: list = None, config: dict = None):
        if question is None:
            # Resuming interrupted execution
            return self.app.stream(None, config=config)
            
        if chat_history is None:
            chat_history = []
        print(f"\n[Starting AegisRAG Workflow Stream] Question: {question}")
        initial_state = {
            "question": question,
            "chat_history": chat_history,
            "documents": [],
            "draft_answer": "",
            "critique": "",
            "is_hallucination": False,
            "needs_rag": True,
            "iterations": 0,
            "final_answer": {}
        }
        return self.app.stream(initial_state, config=config)
