"""
MetaInsight Backend API
FastAPI application for multi-agent document intelligence.

Features:
- Document analysis with 4 agents (Orchestrator, Summarizer, Task Master, Risk Detector)
- Individual agent endpoints
- File upload support
- Async processing
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MetaInsight API",
    description="""
    ## Multi-Agent Document Intelligence System
    
    MetaInsight uses 4 specialized AI agents to analyze documents:
    
    1. **Orchestrator** - Coordinates all agents, manages retrieval and routing
    2. **Summarizer** - Generates document summaries, extracts key decisions
    3. **Task Master** - Extracts actionable tasks, deadlines, assignments
    4. **Risk Detector** - Identifies risks, assumptions, uncertainties, conflicts
    
    ### Features
    - Full document analysis pipeline
    - Individual agent endpoints
    - File upload support (TXT, MD)
    - Async job processing
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request model for document analysis."""
    document: str = Field(..., description="The full text document to analyze", min_length=10)
    chunk_size: int = Field(500, description="Size of each text chunk", ge=100, le=2000)
    overlap: int = Field(100, description="Overlap between chunks", ge=0, le=500)
    top_k: int = Field(5, description="Number of chunks to retrieve per agent", ge=1, le=20)
    agents: List[str] = Field(
        default=["summary", "action", "risk"],
        description="Which agents to run: 'summary', 'action', 'risk'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document": "Project meeting notes: We discussed the new feature implementation...",
                "chunk_size": 500,
                "overlap": 100,
                "top_k": 5,
                "agents": ["summary", "action", "risk"]
            }
        }


class AnalysisResponse(BaseModel):
    """Response model for document analysis."""
    job_id: str
    status: str
    summary: Optional[Dict[str, Any]] = None
    actions: Optional[Dict[str, Any]] = None
    risks: Optional[Dict[str, Any]] = None
    success: bool
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None
    timestamp: str


class RiskAnalysisRequest(BaseModel):
    """Request model for risk-only analysis."""
    document: str = Field(..., description="Document text to analyze for risks", min_length=10)
    chunk_size: int = Field(500, ge=100, le=2000)
    overlap: int = Field(100, ge=0, le=500)
    top_k: int = Field(5, ge=1, le=20)
    include_cross_agent_data: bool = Field(
        default=False,
        description="Include Task Master & Summarizer data for enrichment"
    )


class TaskAnalysisRequest(BaseModel):
    """Request model for task extraction."""
    document: str = Field(..., description="Document text to analyze", min_length=10)
    chunk_size: int = Field(500, ge=100, le=2000)
    overlap: int = Field(100, ge=0, le=500)
    top_k: int = Field(5, ge=1, le=20)


class JobStatus(BaseModel):
    """Job status model for async processing."""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int  # 0-100
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# In-memory job storage (replace with Redis/DB in production)
jobs: Dict[str, JobStatus] = {}


# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "service": "MetaInsight API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs",
        "agents": ["orchestrator", "summarizer", "task_master", "risk_detector"]
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check for all components."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "healthy",
            "vector_store": "healthy",
            "agents": {
                "orchestrator": "available",
                "summarizer": "available",
                "task_master": "available",
                "risk_detector": "available"
            }
        }
    }


@app.get("/agents", tags=["Agents"])
async def list_agents():
    """List all available agents and their capabilities."""
    return {
        "agents": [
            {
                "id": "orchestrator",
                "name": "Orchestrator Agent",
                "description": "Coordinates all agents, manages document chunking and retrieval",
                "capabilities": ["chunking", "embedding", "retrieval", "routing", "aggregation"]
            },
            {
                "id": "summarizer",
                "name": "Summary Agent",
                "description": "Generates document summaries, extracts key decisions and goals",
                "capabilities": ["summarization", "key_extraction", "decision_identification"]
            },
            {
                "id": "task_master",
                "name": "Task Master Agent",
                "description": "Extracts actionable tasks, deadlines, and assignments",
                "capabilities": ["task_extraction", "deadline_detection", "assignment_mapping", "dependency_analysis"]
            },
            {
                "id": "risk_detector",
                "name": "Risk Detection Agent",
                "description": "Identifies risks, assumptions, uncertainties, and conflicts",
                "capabilities": [
                    "risk_identification",
                    "assumption_detection",
                    "uncertainty_analysis",
                    "conflict_detection",
                    "quantitative_scoring"
                ]
            }
        ]
    }


# ============================================================================
# MAIN ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_document(request: AnalysisRequest):
    """
    Analyze a document using the multi-agent pipeline.
    
    This is the main endpoint that runs the full analysis:
    1. Document chunking with overlap
    2. Vector embedding and storage
    3. Adaptive retrieval per agent
    4. Parallel agent execution
    5. Result aggregation
    """
    import time
    start_time = time.time()
    job_id = str(uuid.uuid4())[:8]
    
    try:
        from app.orchestrator import run_pipeline
        
        logger.info(f"[{job_id}] Starting analysis for document ({len(request.document)} chars)")
        
        result = run_pipeline(
            document=request.document,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            top_k=request.top_k
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"[{job_id}] Analysis completed in {processing_time}ms")
        
        return AnalysisResponse(
            job_id=job_id,
            status="completed",
            summary=result.get("summary"),
            actions=result.get("actions"),
            risks=result.get("risks"),
            success=result.get("success", False),
            error=result.get("error"),
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"[{job_id}] Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/async", tags=["Analysis"])
async def analyze_document_async(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Start async document analysis.
    
    Returns immediately with a job_id that can be used to poll for results.
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Create job entry
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        created_at=datetime.utcnow().isoformat(),
        updated_at=datetime.utcnow().isoformat()
    )
    
    # Add background task
    background_tasks.add_task(
        run_analysis_job,
        job_id,
        request.document,
        request.chunk_size,
        request.overlap,
        request.top_k
    )
    
    return {"job_id": job_id, "status": "pending", "poll_url": f"/jobs/{job_id}"}


async def run_analysis_job(job_id: str, document: str, chunk_size: int, overlap: int, top_k: int):
    """Background task for async analysis."""
    try:
        jobs[job_id].status = "processing"
        jobs[job_id].progress = 10
        jobs[job_id].updated_at = datetime.utcnow().isoformat()
        
        from app.orchestrator import run_pipeline
        
        result = run_pipeline(
            document=document,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k
        )
        
        jobs[job_id].status = "completed"
        jobs[job_id].progress = 100
        jobs[job_id].result = result
        jobs[job_id].updated_at = datetime.utcnow().isoformat()
        
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        jobs[job_id].updated_at = datetime.utcnow().isoformat()


@app.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get the status of an async analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


# ============================================================================
# INDIVIDUAL AGENT ENDPOINTS
# ============================================================================

@app.post("/analyze/risks", tags=["Risk Detection"])
async def analyze_risks(request: RiskAnalysisRequest):
    """
    Run Risk Detection Agent only.
    
    Performs 3-phase risk analysis:
    - Phase 1: Linguistic analysis (keyword detection)
    - Phase 2: Contextual analysis (deep reasoning)
    - Phase 3: Cross-agent enrichment (if enabled)
    
    Returns comprehensive risk analysis with quantitative scoring.
    """
    import time
    start_time = time.time()
    
    try:
        from app.agents.risk_agent import run_risk_agent
        from app.chunking import chunk_document
        from app.memory import VectorStore
        
        # Chunk document
        chunks = chunk_document(
            request.document,
            chunk_size=request.chunk_size,
            overlap=request.overlap
        )
        
        # Create vector store and retrieve
        store = VectorStore()
        store.add_documents(chunks)
        
        risk_query = "risks uncertainties blockers issues challenges problems concerns assumptions"
        risk_chunks = store.retrieve(risk_query, top_k=request.top_k)
        
        # Format context
        context = "\n---\n".join([f"[Chunk {i+1}]: {c}" for i, c in enumerate(risk_chunks)])
        
        # Run risk agent
        result = run_risk_agent(context)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "agent": "risk_detector",
            "risks": result,
            "success": True,
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Risk analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/tasks", tags=["Task Master"])
async def analyze_tasks(request: TaskAnalysisRequest):
    """
    Run Task Master Agent only.
    
    Extracts:
    - Actionable tasks
    - Deadlines and timelines
    - Assignments and responsibilities
    - Task dependencies
    """
    import time
    start_time = time.time()
    
    try:
        from app.agents.action_agent import run_action_agent
        from app.chunking import chunk_document
        from app.memory import VectorStore
        
        # Chunk document
        chunks = chunk_document(
            request.document,
            chunk_size=request.chunk_size,
            overlap=request.overlap
        )
        
        # Create vector store and retrieve
        store = VectorStore()
        store.add_documents(chunks)
        
        action_query = "tasks action items deadlines responsibilities assignments deliverables"
        action_chunks = store.retrieve(action_query, top_k=request.top_k)
        
        # Format context
        context = "\n---\n".join([f"[Chunk {i+1}]: {c}" for i, c in enumerate(action_chunks)])
        
        # Run action agent
        result = run_action_agent(context)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "agent": "task_master",
            "tasks": result,
            "success": True,
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Task analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/summary", tags=["Summarizer"])
async def analyze_summary(request: TaskAnalysisRequest):
    """
    Run Summary Agent only.
    
    Generates:
    - Executive summary
    - Key decisions
    - Main goals and objectives
    - Outcomes and conclusions
    """
    import time
    start_time = time.time()
    
    try:
        from app.agents.summary_agent import run_summary_agent
        from app.chunking import chunk_document
        from app.memory import VectorStore
        
        # Chunk document
        chunks = chunk_document(
            request.document,
            chunk_size=request.chunk_size,
            overlap=request.overlap
        )
        
        # Create vector store and retrieve
        store = VectorStore()
        store.add_documents(chunks)
        
        summary_query = "overall goals objectives decisions outcomes strategic direction summary"
        summary_chunks = store.retrieve(summary_query, top_k=request.top_k)
        
        # Format context
        context = "\n---\n".join([f"[Chunk {i+1}]: {c}" for i, c in enumerate(summary_chunks)])
        
        # Run summary agent
        result = run_summary_agent(context)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "agent": "summarizer",
            "summary": result,
            "success": True,
            "processing_time_ms": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Summary analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FILE UPLOAD ENDPOINTS
# ============================================================================

@app.post("/upload", tags=["File Upload"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document file for analysis.
    
    Supported formats: .txt, .md
    Returns document text that can be used with /analyze endpoint.
    """
    allowed_extensions = [".txt", ".md", ".text"]
    
    # Check file extension
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
            "document": text,
            "message": "File uploaded successfully. Use the 'document' field with /analyze endpoint."
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not decode file. Ensure it's UTF-8 encoded text.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/analyze", tags=["File Upload"])
async def upload_and_analyze(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    overlap: int = 100,
    top_k: int = 5
):
    """
    Upload a document and immediately analyze it.
    
    Combines file upload and full analysis in one request.
    """
    # First upload
    upload_result = await upload_document(file)
    document_text = upload_result["document"]
    
    # Then analyze
    request = AnalysisRequest(
        document=document_text,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k
    )
    
    return await analyze_document(request)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("MetaInsight API starting up...")
    logger.info("4 agents available: Orchestrator, Summarizer, Task Master, Risk Detector")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("MetaInsight API shutting down...")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
