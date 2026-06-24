from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class AegisRAGState(TypedDict):
    """
    State for the AegisRAG workflow.
    """
    question: str
    chat_history: List[BaseMessage]
    documents: List[Document]
    draft_answer: str
    critique: str
    is_hallucination: bool
    needs_rag: bool
    final_answer: dict # Structured output
    iterations: int
