import pytest
from src.agents.graph import AegisRAGWorkflow

def test_graph_routing_rewrite():
    workflow = AegisRAGWorkflow()
    state_rewrite = {"is_hallucination": True, "iterations": 1}
    route = workflow._decide_next_step(state_rewrite)
    assert route == "rewrite"

def test_graph_routing_end_on_success():
    workflow = AegisRAGWorkflow()
    state_success = {"is_hallucination": False, "iterations": 1}
    route = workflow._decide_next_step(state_success)
    assert route == "end"

def test_graph_routing_end_on_max_iterations():
    workflow = AegisRAGWorkflow(max_iterations=3)
    state_max = {"is_hallucination": True, "iterations": 3}
    route = workflow._decide_next_step(state_max)
    assert route == "end"
