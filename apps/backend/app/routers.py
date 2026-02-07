"""
API Router Module
Organizes endpoints into separate routers for better code organization.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from typing import Dict, Any, List
from datetime import datetime
import time
import uuid

from app.config import settings

# Create routers
analysis_router = APIRouter(prefix="/analyze", tags=["Analysis"])
agents_router = APIRouter(prefix="/agents", tags=["Agents"])
files_router = APIRouter(prefix="/files", tags=["Files"])


# ============================================================================
# ANALYSIS ROUTES
# ============================================================================

@analysis_router.post("/full")
async def full_analysis(document: str, agents: List[str] = ["summary", "action", "risk"]):
    """Run full analysis with selected agents."""
    from app.orchestrator import run_pipeline
    
    start_time = time.time()
    job_id = str(uuid.uuid4())[:8]
    
    try:
        result = run_pipeline(
            document=document,
            chunk_size=settings.default_chunk_size,
            overlap=settings.default_overlap,
            top_k=settings.default_top_k
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "result": result,
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# AGENT INFO ROUTES
# ============================================================================

@agents_router.get("/")
async def list_all_agents():
    """List all available agents."""
    return {
        "agents": [
            {"id": "orchestrator", "status": "active"},
            {"id": "summarizer", "status": "active"},
            {"id": "task_master", "status": "active"},
            {"id": "risk_detector", "status": "active"}
        ]
    }


@agents_router.get("/{agent_id}")
async def get_agent_info(agent_id: str):
    """Get detailed info about a specific agent."""
    agents_info = {
        "orchestrator": {
            "id": "orchestrator",
            "name": "Orchestrator Agent",
            "description": "Coordinates all agents and manages the pipeline",
            "version": "1.0.0"
        },
        "summarizer": {
            "id": "summarizer",
            "name": "Summary Agent",
            "description": "Generates document summaries",
            "version": "1.0.0"
        },
        "task_master": {
            "id": "task_master",
            "name": "Task Master Agent",
            "description": "Extracts tasks and action items",
            "version": "1.0.0"
        },
        "risk_detector": {
            "id": "risk_detector",
            "name": "Risk Detection Agent",
            "description": "Identifies risks and uncertainties",
            "version": "1.0.0"
        }
    }
    
    if agent_id not in agents_info:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    
    return agents_info[agent_id]


# ============================================================================
# FILE ROUTES
# ============================================================================

@files_router.post("/parse")
async def parse_file(file: UploadFile = File(...)):
    """Parse uploaded file and return text content."""
    allowed_extensions = [".txt", ".md", ".text"]
    
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        return {
            "filename": file.filename,
            "size_bytes": len(content),
            "char_count": len(text),
            "content": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
