from typing import TypedDict, List
from langchain_core.documents import Document

class AegisRAGState(TypedDict):
    """
    State for the AegisRAG workflow.
    """
    question: str
    documents: List[Document]
    draft_answer: str
    critique: str
    is_hallucination: bool
    final_answer: dict # Structured output
    iterations: int
