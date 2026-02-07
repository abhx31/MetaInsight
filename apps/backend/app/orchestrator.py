"""
Orchestrator: The main pipeline coordinator for multi-agent document intelligence.

This module now uses LangGraph for state-based orchestration.
See orchestrator_langgraph.py for the full implementation.
"""

# Re-export from LangGraph orchestrator
from app.orchestrator_langgraph import (
    run_pipeline,
    process_document,
    get_pipeline_graph,
    create_pipeline_graph,
    get_graph_mermaid,
    PipelineState,
)

__all__ = [
    "run_pipeline",
    "process_document",
    "get_pipeline_graph",
    "create_pipeline_graph",
    "get_graph_mermaid",
    "PipelineState",
]
