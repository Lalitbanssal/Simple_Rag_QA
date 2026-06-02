from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import get_settings
from app.utils.logger import get_logger,LoggerMixin

@lru_cache(maxsize=None)
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Get embedding model instance."""
    settings = get_settings()
    logger = get_logger(__name__)
    logger.info(f"Initializing embeddings model: {settings.embedding_model}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.googleapikey,
    )

    logger.info("Embeddings model initialized successfully")
    return embeddings

class EmbeddingGenerator(LoggerMixin):
    def __init__(self):
        """Initialize embedding service."""
        self.settings = get_settings()
        self.embeddings = get_embedding_model()
        self.model_name = self.settings.embedding_model
        self.logger.info(f"EmbeddingGenerator initialized with model: {self.model_name}")
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text

        Returns:
            Embedding vector as list of floats
        """
        logger.debug(f"Generating embedding for query: {text[:50]}...")
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of document texts

        Returns:
            List of embedding vectors
        """
        logger.debug(f"Generating embeddings for {len(texts)} documents")
        return self.embeddings.embed_documents(texts)

