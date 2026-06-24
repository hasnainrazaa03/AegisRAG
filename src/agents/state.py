from typing import TypedDict, List
from langchain_core.documents import Document

class AgenticRAGState(TypedDict):
    """
    State for the Agentic RAG workflow.
    """
    question: str
    documents: List[Document]
    draft_answer: str
    critique: str
    is_hallucination: bool
    final_answer: dict # Structured output
    iterations: int
