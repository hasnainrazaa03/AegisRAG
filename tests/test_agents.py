import pytest
from unittest.mock import MagicMock
from src.agents.synthesizer import SynthesizerAgent
from src.agents.critic import CriticAgent
from langchain_core.documents import Document

def test_synthesizer_empty_context():
    agent = SynthesizerAgent()
    state = {
        "question": "What is the structural tolerance?",
        "documents": []
    }
    
    result = agent.invoke(state)
    assert "No relevant documents" in result["draft_answer"] or "could not find any relevant information" in result["draft_answer"]

def test_synthesizer_with_context():
    agent = SynthesizerAgent()
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = "The tolerance is 5000 PSI."
    
    state = {
        "question": "What is the structural tolerance?",
        "documents": [Document(page_content="5000 PSI limit.")]
    }
    
    result = agent.invoke(state)
    assert "5000 PSI" in result["draft_answer"]

def test_critic_valid_json_no_hallucination():
    agent = CriticAgent()
    agent.chain = MagicMock()
    agent.chain.invoke.return_value = {"is_hallucination": False, "critique": "Looks good."}
    
    state = {
        "draft_answer": "The tolerance is 5000 PSI.",
        "documents": [Document(page_content="5000 PSI limit.")],
        "iterations": 1
    }
    
    result = agent.invoke(state)
    assert result["is_hallucination"] is False
    assert result["iterations"] == 2

def test_critic_malformed_json_fallback():
    agent = CriticAgent()
    agent.chain = MagicMock()
    # Simulate chain raising a parse error
    agent.chain.invoke.side_effect = Exception("JSONDecodeError")
    
    state = {
        "draft_answer": "The tolerance is 5000 PSI.",
        "documents": [Document(page_content="5000 PSI limit.")],
        "iterations": 1
    }
    
    result = agent.invoke(state)
    # The new robust behavior should assume hallucination and force rewrite
    assert result["is_hallucination"] is True
    assert "failed to parse" in result["critique"].lower()
