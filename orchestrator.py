"""
Orchestrator: The main pipeline coordinator for multi-agent document intelligence.

This module orchestrates the entire document processing pipeline:
1. Chunking: Split long documents into overlapping segments
2. Embedding: Convert chunks to vector embeddings and store in FAISS
3. Adaptive Retrieval: Retrieve relevant context for each specialized agent
4. Agent Execution: Run each agent with its customized context
5. Aggregation: Combine agent outputs into structured JSON

The orchestrator ensures agents remain decoupled from the vector store implementation.
"""

from typing import Dict, Any, Optional
import logging

from app.chunking import chunk_document
from app.memory import VectorStore
from app.agents.summary_agent import run_summary_agent
from app.agents.action_agent import run_action_agent
from app.agents.risk_agent import run_risk_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(
    document: str,
    chunk_size: int = 500,
    overlap: int = 100,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Execute the full multi-agent document intelligence pipeline.
    
    This is the main entry point that coordinates all components:
    - Document chunking with overlap
    - Vector store creation and embedding
    - Adaptive retrieval per agent type
    - Parallel agent execution
    - Result aggregation
    
    Args:
        document: The input document text to analyze
        chunk_size: Size of each chunk in characters (default: 500)
        overlap: Overlap between chunks in characters (default: 100)
        top_k: Number of relevant chunks to retrieve per agent (default: 5)
    
    Returns:
        Dictionary containing structured output from all agents:
        {
            "summary": str or dict,
            "actions": list or dict,
            "risks": list or dict,
            "success": bool,
            "error": Optional[str]
        }
    
    Example:
        >>> document = "Project meeting notes: We discussed the new feature..."
        >>> result = run_pipeline(document)
        >>> print(result['summary'])
        >>> print(result['actions'])
        >>> print(result['risks'])
    """
    result: Dict[str, Any] = {
        "summary": None,
        "actions": None,
        "risks": None,
        "success": False,
        "error": None
    }
    
    try:
        # ============================================================
        # STEP 1: CHUNKING
        # Break the document into overlapping chunks for better context preservation
        # ============================================================
        logger.info(f"Starting pipeline for document of length {len(document)} characters")
        
        if not document or not document.strip():
            raise ValueError("Document cannot be empty")
        
        logger.info(f"Chunking document with chunk_size={chunk_size}, overlap={overlap}")
        chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
        logger.info(f"Created {len(chunks)} chunks")
        
        if not chunks:
            raise ValueError("Chunking produced no results")
        
        # ============================================================
        # STEP 2: VECTOR STORE CREATION
        # Convert chunks to embeddings and store in FAISS index
        # This happens ONCE per document to avoid redundant computation
        # ============================================================
        logger.info("Initializing vector store and creating embeddings")
        store = VectorStore()
        store.add_documents(chunks)
        logger.info("Vector store initialized successfully")
        
        # ============================================================
        # STEP 3: ADAPTIVE RETRIEVAL
        # Each agent gets context tailored to its specific purpose
        # Queries are designed to retrieve the most relevant chunks
        # ============================================================
        logger.info("Performing adaptive retrieval for each agent")
        
        # Summary Agent: Focus on high-level goals, decisions, and outcomes
        summary_query = "overall goals objectives decisions outcomes strategic direction"
        summary_chunks = store.retrieve(summary_query, top_k=top_k)
        summary_context = " ".join(summary_chunks)
        logger.info(f"Retrieved {len(summary_chunks)} chunks for summary agent")
        
        # Action Agent: Focus on tasks, deadlines, responsibilities, and action items
        action_query = "tasks action items deadlines responsibilities assignments deliverables"
        action_chunks = store.retrieve(action_query, top_k=top_k)
        action_context = " ".join(action_chunks)
        logger.info(f"Retrieved {len(action_chunks)} chunks for action agent")
        
        # Risk Agent: Focus on uncertainties, blockers, issues, and challenges
        risk_query = "risks uncertainties blockers issues challenges problems concerns"
        risk_chunks = store.retrieve(risk_query, top_k=top_k)
        risk_context = " ".join(risk_chunks)
        logger.info(f"Retrieved {len(risk_chunks)} chunks for risk agent")
        
        # ============================================================
        # STEP 4: AGENT EXECUTION (PARALLEL)
        # Run specialized agents in parallel to minimize pipeline latency.
        # Each agent is isolated to prevent one failure from blocking others.
        # ============================================================
        logger.info("Running agents in parallel with retrieved context")
        
        from concurrent.futures import ThreadPoolExecutor
        
        # Format contexts with clear separators to help agents distinguish between chunks
        def format_context(chunks: list) -> str:
            if not chunks:
                return ""
            return "\n---\n".join([f"[Source Chunk {i+1}]: {chunk}" for i, chunk in enumerate(chunks)])

        s_ctx = format_context(summary_chunks)
        a_ctx = format_context(action_chunks)
        r_ctx = format_context(risk_chunks)

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Schedule agent tasks
            summary_task = executor.submit(run_summary_agent, s_ctx)
            action_task = executor.submit(run_action_agent, a_ctx)
            risk_task = executor.submit(run_risk_agent, r_ctx)

            # Retrieve results with individual error handling
            try:
                result["summary"] = summary_task.result()
            except Exception as e:
                logger.error(f"Summary agent failed: {str(e)}")
                result["summary"] = {"error": f"Internal agent error: {str(e)}"}

            try:
                result["actions"] = action_task.result()
            except Exception as e:
                logger.error(f"Action agent failed: {str(e)}")
                result["actions"] = {"error": f"Internal agent error: {str(e)}"}

            try:
                result["risks"] = risk_task.result()
            except Exception as e:
                logger.error(f"Risk agent failed: {str(e)}")
                result["risks"] = {"error": f"Internal agent error: {str(e)}"}
        
        # ============================================================
        # STEP 5: RESULT AGGREGATION
        # Combine all outputs into structured JSON
        # ============================================================
        result["success"] = True
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        # Catch any unexpected errors and return them gracefully
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        result["error"] = str(e)
        result["success"] = False
    
    return result


def process_document(document: str) -> Dict[str, Any]:
    """
    Convenience wrapper for run_pipeline with default parameters.
    
    Args:
        document: The input document text
        
    Returns:
        Processed results from all agents
    """
    return run_pipeline(document)
