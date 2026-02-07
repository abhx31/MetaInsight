"""
Application Configuration
Environment variables and settings management.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    app_name: str = "MetaInsight API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS Settings
    cors_origins: str = "*"
    
    # OpenAI Settings (for LLM calls)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.1
    
    # Vector Store Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_directory: Optional[str] = None
    
    # Document Processing Settings
    default_chunk_size: int = 500
    default_overlap: int = 100
    default_top_k: int = 5
    max_document_size: int = 1000000  # 1MB
    
    # Agent Settings
    agent_timeout: int = 60  # seconds
    parallel_agents: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()
