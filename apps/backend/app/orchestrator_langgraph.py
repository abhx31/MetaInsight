"""
LangGraph Orchestrator: State-based multi-agent document intelligence pipeline.

Uses LangGraph for:
- Typed state management across nodes
- Parallel agent execution
- Conditional routing and error handling
- Built-in checkpointing support
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from operator import add
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from app.chunking import chunk_document
from app.memory import VectorStore
from app.agents.summary_agent import run_summary_agent
from app.agents.action_agent import run_action_agent
from app.agents.risk_agent import run_risk_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class PipelineState(TypedDict):
    """State that flows through the LangGraph pipeline."""
    # Input
    document: str
    chunk_size: int
    overlap: int
    top_k: int
    
    # Processing state
    chunks: List[str]
    summary_context: str
    action_context: str
    risk_context: str
    
    # Agent outputs
    summary_result: Optional[Dict[str, Any]]
    action_result: Optional[Dict[str, Any]]
    risk_result: Optional[Dict[str, Any]]
    
    # Final output
    success: bool
    error: Optional[str]
    
    # Metadata
    messages: Annotated[List[str], add]  # Accumulates log messages


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def chunking_node(state: PipelineState) -> Dict[str, Any]:
    """Split document into overlapping chunks."""
    logger.info("🔪 Chunking node: Starting...")
    
    document = state["document"]
    chunk_size = state.get("chunk_size", 500)
    overlap = state.get("overlap", 100)
    
    if not document or not document.strip():
        return {
            "error": "Document cannot be empty",
            "success": False,
            "messages": ["ERROR: Empty document provided"]
        }
    
    chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    
    if not chunks:
        return {
            "error": "Chunking produced no results",
            "success": False,
            "messages": ["ERROR: Chunking failed"]
        }
    
    logger.info(f"✅ Created {len(chunks)} chunks")
    return {
        "chunks": chunks,
        "messages": [f"Created {len(chunks)} chunks from document ({len(document)} chars)"]
    }


def embedding_node(state: PipelineState) -> Dict[str, Any]:
    """Create embeddings and store in vector store, then retrieve context for each agent."""
    logger.info("🧠 Embedding node: Creating vector store...")
    
    chunks = state.get("chunks", [])
    top_k = state.get("top_k", 5)
    
    if not chunks:
        return {
            "error": "No chunks to embed",
            "success": False,
            "messages": ["ERROR: No chunks available for embedding"]
        }
    
    try:
        # Create vector store
        store = VectorStore()
        store.add_documents(chunks)
        
        # Adaptive retrieval for each agent
        summary_query = "overall goals objectives decisions outcomes strategic direction"
        action_query = "tasks action items deadlines responsibilities assignments deliverables"
        risk_query = "risks uncertainties blockers issues challenges problems concerns"
        
        summary_chunks = store.retrieve(summary_query, top_k=top_k)
        action_chunks = store.retrieve(action_query, top_k=top_k)
        risk_chunks = store.retrieve(risk_query, top_k=top_k)
        
        # Format contexts
        def format_context(chunks_list: list) -> str:
            if not chunks_list:
                return ""
            return "\n---\n".join([f"[Source Chunk {i+1}]: {chunk}" for i, chunk in enumerate(chunks_list)])
        
        logger.info(f"✅ Retrieved contexts: summary={len(summary_chunks)}, action={len(action_chunks)}, risk={len(risk_chunks)}")
        
        return {
            "summary_context": format_context(summary_chunks),
            "action_context": format_context(action_chunks),
            "risk_context": format_context(risk_chunks),
            "messages": [f"Vector store created, retrieved {top_k} chunks per agent"]
        }
        
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        return {
            "error": f"Embedding failed: {str(e)}",
            "success": False,
            "messages": [f"ERROR: Embedding failed - {str(e)}"]
        }


def summary_agent_node(state: PipelineState) -> Dict[str, Any]:
    """Run the Summary Agent."""
    logger.info("📝 Summary Agent: Running...")
    
    context = state.get("summary_context", "")
    
    try:
        result = run_summary_agent(context)
        logger.info("✅ Summary Agent completed")
        return {
            "summary_result": result,
            "messages": ["Summary Agent completed successfully"]
        }
    except Exception as e:
        logger.error(f"Summary Agent failed: {str(e)}")
        return {
            "summary_result": {"error": str(e)},
            "messages": [f"Summary Agent failed: {str(e)}"]
        }


def action_agent_node(state: PipelineState) -> Dict[str, Any]:
    """Run the Action/Task Master Agent."""
    logger.info("✅ Action Agent: Running...")
    
    context = state.get("action_context", "")
    
    try:
        result = run_action_agent(context)
        logger.info("✅ Action Agent completed")
        return {
            "action_result": result,
            "messages": ["Action Agent completed successfully"]
        }
    except Exception as e:
        logger.error(f"Action Agent failed: {str(e)}")
        return {
            "action_result": {"error": str(e)},
            "messages": [f"Action Agent failed: {str(e)}"]
        }


def risk_agent_node(state: PipelineState) -> Dict[str, Any]:
    """Run the Risk Detection Agent."""
    logger.info("⚠️ Risk Agent: Running...")
    
    context = state.get("risk_context", "")
    
    try:
        result = run_risk_agent(context)
        logger.info("✅ Risk Agent completed")
        return {
            "risk_result": result,
            "messages": ["Risk Agent completed successfully"]
        }
    except Exception as e:
        logger.error(f"Risk Agent failed: {str(e)}")
        return {
            "risk_result": {"error": str(e)},
            "messages": [f"Risk Agent failed: {str(e)}"]
        }


def aggregation_node(state: PipelineState) -> Dict[str, Any]:
    """Aggregate all agent results into final output."""
    logger.info("🔄 Aggregation node: Combining results...")
    
    # Check if any critical errors occurred
    has_error = state.get("error") is not None
    
    return {
        "success": not has_error,
        "messages": ["Pipeline completed - results aggregated"]
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_pipeline_graph() -> CompiledStateGraph:
    """Create and compile the LangGraph pipeline."""
    
    # Create the graph
    graph = StateGraph(PipelineState)
    
    # Add nodes
    graph.add_node("chunking", chunking_node)
    graph.add_node("embedding", embedding_node)
    graph.add_node("summary_agent", summary_agent_node)
    graph.add_node("action_agent", action_agent_node)
    graph.add_node("risk_agent", risk_agent_node)
    graph.add_node("aggregation", aggregation_node)
    
    # Define edges
    # START -> Chunking -> Embedding
    graph.add_edge(START, "chunking")
    graph.add_edge("chunking", "embedding")
    
    # Embedding -> All agents (parallel fan-out)
    graph.add_edge("embedding", "summary_agent")
    graph.add_edge("embedding", "action_agent")
    graph.add_edge("embedding", "risk_agent")
    
    # All agents -> Aggregation (parallel fan-in)
    graph.add_edge("summary_agent", "aggregation")
    graph.add_edge("action_agent", "aggregation")
    graph.add_edge("risk_agent", "aggregation")
    
    # Aggregation -> END
    graph.add_edge("aggregation", END)
    
    # Compile the graph
    return graph.compile()


# Global compiled graph instance
_pipeline_graph: Optional[CompiledStateGraph] = None


def get_pipeline_graph() -> CompiledStateGraph:
    """Get or create the compiled pipeline graph."""
    global _pipeline_graph
    if _pipeline_graph is None:
        _pipeline_graph = create_pipeline_graph()
    return _pipeline_graph


# ============================================================================
# PUBLIC API
# ============================================================================

def run_pipeline(
    document: str,
    chunk_size: int = 500,
    overlap: int = 100,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Execute the full multi-agent document intelligence pipeline using LangGraph.
    
    Args:
        document: The input document text to analyze
        chunk_size: Size of each chunk in characters (default: 500)
        overlap: Overlap between chunks in characters (default: 100)
        top_k: Number of relevant chunks to retrieve per agent (default: 5)
    
    Returns:
        Dictionary containing structured output from all agents
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting LangGraph Pipeline")
    logger.info("=" * 60)
    
    # Initialize state
    initial_state: PipelineState = {
        "document": document,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k,
        "chunks": [],
        "summary_context": "",
        "action_context": "",
        "risk_context": "",
        "summary_result": None,
        "action_result": None,
        "risk_result": None,
        "success": False,
        "error": None,
        "messages": []
    }
    
    try:
        # Get compiled graph
        graph = get_pipeline_graph()
        
        # Run the pipeline
        final_state = graph.invoke(initial_state)
        
        logger.info("=" * 60)
        logger.info("✅ Pipeline completed successfully")
        logger.info("=" * 60)
        
        # Format output
        return {
            "summary": final_state.get("summary_result"),
            "actions": final_state.get("action_result"),
            "risks": final_state.get("risk_result"),
            "success": final_state.get("success", False) and final_state.get("error") is None,
            "error": final_state.get("error"),
            "pipeline_messages": final_state.get("messages", [])
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return {
            "summary": None,
            "actions": None,
            "risks": None,
            "success": False,
            "error": str(e),
            "pipeline_messages": [f"Pipeline exception: {str(e)}"]
        }


def process_document(document: str) -> Dict[str, Any]:
    """Convenience wrapper for run_pipeline with default parameters."""
    return run_pipeline(document)


# ============================================================================
# VISUALIZATION (for debugging)
# ============================================================================

def get_graph_mermaid() -> str:
    """Get Mermaid diagram of the pipeline graph."""
    return """
    graph TD
        START((Start)) --> chunking[🔪 Chunking]
        chunking --> embedding[🧠 Embedding]
        embedding --> summary[📝 Summary Agent]
        embedding --> action[✅ Action Agent]
        embedding --> risk[⚠️ Risk Agent]
        summary --> aggregation[🔄 Aggregation]
        action --> aggregation
        risk --> aggregation
        aggregation --> END((End))
    """
