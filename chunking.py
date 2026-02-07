"""
Document chunking utilities for splitting long documents into overlapping segments.

Chunking is essential for processing long documents that exceed LLM context windows
and for creating semantic search indices.
"""

from typing import List


def chunk_document(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 100
) -> List[str]:
    """
    Split a document into overlapping chunks of approximately equal size.
    
    Overlapping chunks ensure that context isn't lost at chunk boundaries,
    which is important for semantic search and document understanding.
    
    Args:
        text: The input document text to be chunked
        chunk_size: Maximum size of each chunk in characters (default: 500)
        overlap: Number of overlapping characters between chunks (default: 100)
                This creates continuity between adjacent chunks.
    
    Returns:
        List of text chunks with overlap
        
    Raises:
        ValueError: If parameters are invalid
        
    Example:
        >>> text = "This is a long document..." * 100
        >>> chunks = chunk_document(text, chunk_size=500, overlap=100)
        >>> # Each chunk will be ~500 chars with 100 chars overlap
    """
    # Input validation
    if not text or not text.strip():
        return []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
        
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    chunks: List[str] = []
    start = 0
    
    # Ensure text is treated strictly as a string for static analysis
    doc_text: str = str(text)
    text_length = len(doc_text)
    
    # Create chunks with overlap
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = doc_text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        # Move start forward by (chunk_size - overlap)
        # This creates the overlap between consecutive chunks
        start += (chunk_size - overlap)
        
        # Prevent infinite loop if we're at the end
        if end == text_length:
            break
    
    return chunks
