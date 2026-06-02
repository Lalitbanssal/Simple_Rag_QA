"""Vector Store module"""
from functools import lru_cache
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

from app.core.embeddings import get_embedding_model
from app.config import get_settings
from app.utils.logger import get_logger,LoggerMixin

settings = get_settings()
# Embedding dimension for text-embedding-3-small
EMBEDDING_DIMENSION = 1536
@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Get cached Qdrant client instance.

    Returns:
        Configured QdrantClient instance
    """
    logger=get_logger(__name__)
    logger.info(f"Connecting to Qdrant at: {settings.qdrant_url}")

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    logger.info("Qdrant client connected successfully")
    return client

class VectorStore(LoggerMixin):
    """Service for managing vector store operations."""

    def __init__(self, collection_name: str | None = None):
        """Initialize vector store service.

        Args:
            collection_name: Name of the Qdrant collection (default from settings)
        """
        self.collection_name = collection_name or settings.collection_name
        self.client = get_qdrant_client()
        self.embeddings = get_embeddings()
        self._ensure_collection()
        # Initialize LangChain Qdrant vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        self.logger.info(f"VectorStore initialized with collection: {self.collection_name}")
    def _ensure_collection(self)-> None:
        """Ensure collection exists with proper configuration."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(
                f"Collection '{self.collection_name}' exists with "
                f"{collection_info.points_count} points"
            )
        except UnexpectedResponse:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")

    def add_documents(self,documents: list[Document])-> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs
        """
        if not documents:
            self.logger.warning("No documents to add")
            return []
        logger.info(f"Adding {len(documents)} documents to collection")

        
        # Generate unique IDs for each document
        ids = [str(uuid4()) for _ in documents]

        # Add to vector store
        self.vector_store.add_documents(documents, ids=ids)

        logger.info(f"Successfully added {len(documents)} documents")
        return ids
    
    def search(self,query: str,k: int | None = None,) -> list[Document]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return (default from settings)

        Returns:
            List of similar Document objects
        """
        k = k or settings.retrieval_k
        logger.debug(f"Searching for: {query[:50]}... (k={k})")

        results = self.vector_store.similarity_search(query, k=k)

        logger.debug(f"Found {len(results)} results")
        return results
    
    def search_by_metadata(self,query: str,k: int | None = None,metadata_filter: dict[str, Any] | None = None) -> list[Document]:
        """Search for similar documents with metadata filtering.

        Args:
            query: Search query
            k: Number of results to return (default from settings)
            metadata_filter: Metadata filter

        Returns:
            List of similar Document objects
        """
        k = k or settings.retrieval_k
        logger.debug(f"Searching for: {query[:50]}... (k={k}, filter={metadata_filter})")

        results = self.vector_store.similarity_search(query, k=k, filter=metadata_filter)

        logger.debug(f"Found {len(results)} results")
        return results
    
    def search_with_score(self,query: str,k: int | None = None) -> list[tuple[Document,float]]:
        """Search for similar documents with score.

        Args:
            query: Search query
            k: Number of results to return (default from settings)


        Returns:
            List of similar Document objects with scores
        """
        k = k or settings.retrieval_k
        logger.debug(f"Searching for: {query[:50]}... (k={k})")

        results = self.vector_store.similarity_search_with_score(query, k=k)

        logger.debug(f"Found {len(results)} results")
        return results
    
    def get_retriever(self,k: int | None = None) -> Any:
        """Get retriever.

        Args:
            k: Number of results to return (default from settings)

        Returns:
            Retriever
        """
        k = k or settings.retrieval_k
        logger.debug(f"Getting retriever with k={k}")

        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def delete_collection(self)-> None:
        """Delete collection."""
        try:
            self.logger.warning(f"Deleting collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Successfully deleted collection")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {str(e)}", exc_info=True)
            raise e
    
    def get_collection_info(self) -> dict:
        """Get collection information.

        Returns:
            Collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            self.logger.info(f"Collection '{self.collection_name}' info: {collection_info}")
            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status.value,
            }
        except UnexpectedResponse:
            return {
                "name": self.collection_name,
                "points_count": 0,
                "indexed_vectors_count": 0,
                "status": "not_found",
            }
        
    def health_check(self)-> bool:
        """Check if Qdrant is healthy."""
        try:
            self.client.get_collection()
            return True
        except Exception as e:
            self.logger.error(f"Vector store health check failed: {e}")
            return False
    
    

