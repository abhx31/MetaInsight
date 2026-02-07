"""
MetaInsight Backend Application
Multi-agent document intelligence system.

Modules:
- agents: Specialized AI agents (Summary, Action, Risk)
- chunking: Document segmentation utilities
- memory: Vector store for semantic retrieval
- orchestrator: Pipeline coordination
- config: Application settings
"""

from .chunking import chunk_document
from .memory import VectorStore
from .config import settings, get_settings

__all__ = [
    "chunk_document",
    "VectorStore",
    "settings",
    "get_settings",
]
