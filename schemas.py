from pydantic import BaseModel, Field

class AnalysisRequest(BaseModel):
    document: str = Field(..., description="The full text document to analyze")
    chunk_size: int = Field(500, description="Size of each text chunk")
    overlap: int = Field(100, description="Overlap between chunks")
    top_k: int = Field(5, description="Number of chunks to retrieve per agent")
