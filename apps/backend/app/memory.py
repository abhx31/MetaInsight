"""
Vector Store Memory Module
Handles document embedding and semantic retrieval using ChromaDB.

This module provides the VectorStore class that:
1. Embeds document chunks using sentence-transformers
2. Stores embeddings in ChromaDB
3. Retrieves semantically similar chunks for agent queries
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for document embeddings and semantic retrieval.
    
    Uses ChromaDB as the backend with sentence-transformer embeddings.
    Designed to be instantiated once per document processing pipeline.
    """
    
    def __init__(self, collection_name: str = "document_chunks"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name for the ChromaDB collection
        """
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_function = None
        self._initialized = False
        
    def _initialize(self):
        """Lazily initialize ChromaDB and embedding function."""
        if self._initialized:
            return
            
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            # Create ephemeral client (in-memory)
            self._client = chromadb.Client()
            
            # Use sentence-transformer for embeddings
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Create or get collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function
            )
            
            self._initialized = True
            logger.info(f"VectorStore initialized with collection '{self.collection_name}'")
            
        except ImportError as e:
            logger.error(f"Failed to import ChromaDB dependencies: {e}")
            logger.info("Falling back to simple keyword matching")
            self._initialized = True  # Mark as initialized to use fallback
            
    def add_documents(self, chunks: List[str], metadata: Optional[List[dict]] = None):
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of text chunks to embed and store
            metadata: Optional metadata for each chunk
        """
        self._initialize()
        
        if not chunks:
            logger.warning("No chunks provided to add_documents")
            return
            
        if self._collection is not None:
            # Generate IDs for chunks
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            # Add to ChromaDB
            self._collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadata if metadata else [{"index": i} for i in range(len(chunks))]
            )
            logger.info(f"Added {len(chunks)} chunks to vector store")
        else:
            # Fallback: store chunks in memory
            self._chunks = chunks
            logger.info(f"Stored {len(chunks)} chunks in memory (fallback mode)")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: Search query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant document chunks
        """
        self._initialize()
        
        if self._collection is not None:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self._collection.count())
                )
                
                documents = results.get("documents", [[]])[0]
                logger.info(f"Retrieved {len(documents)} chunks for query: '{query[:50]}...'")
                return documents
                
            except Exception as e:
                logger.error(f"ChromaDB query failed: {e}")
                return self._fallback_retrieve(query, top_k)
        else:
            return self._fallback_retrieve(query, top_k)
    
    def _fallback_retrieve(self, query: str, top_k: int) -> List[str]:
        """
        Simple keyword-based retrieval fallback.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of matching chunks
        """
        if not hasattr(self, '_chunks') or not self._chunks:
            return []
            
        # Simple keyword matching
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self._chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words & chunk_words)
            scored_chunks.append((score, chunk))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def clear(self):
        """Clear all documents from the store."""
        if self._collection is not None:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function
            )
        if hasattr(self, '_chunks'):
            self._chunks = []
        logger.info("Vector store cleared")
