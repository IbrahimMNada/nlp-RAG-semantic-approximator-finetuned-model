"""
Service for generating and managing vector embeddings.
"""
import logging
from typing import List, Tuple, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ....abstractions.interfaces.embedding_provider_interface import IEmbeddingProvider
from ..entities import ParagraphEmbedding1024, ArticleEmbedding1024

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles embedding generation and storage using a pluggable provider."""
    
    def __init__(self, provider: IEmbeddingProvider):
        self._provider = provider
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed (will be normalized by provider)
            
        Returns:
            Embedding vector
        """
        return await self._provider.generate_embedding(text)
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for txt in texts:
            try:
                embedding = await self.generate_embedding(txt)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                continue
        return embeddings
    
    async def save_paragraph_embeddings(
        self,
        session: AsyncSession,
        article_id: int,
        paragraph_data: List[Tuple[int, str]]
    ) -> List[List[float]]:
        """
        Generate and save paragraph embeddings.
        
        Args:
            session: Database session
            article_id: Article ID
            paragraph_data: List of (paragraph_id, content) tuples
            
        Returns:
            List of generated embedding vectors
        """
        # Clear existing embeddings
        await session.execute(
            text("DELETE FROM paragraph_embeddings_1024 WHERE article_id = :article_id"),
            {"article_id": article_id}
        )
        
        vectors = []
        for paragraph_id, content in paragraph_data:
            try:
                vector = await self.generate_embedding(content)
                vectors.append(vector)
                
                embedding = ParagraphEmbedding1024(
                    article_id=article_id,
                    paragraph_id=paragraph_id,
                    embedding=vector
                )
                session.add(embedding)
                
                logger.debug(f"Saved embedding for paragraph {paragraph_id}")
                
            except Exception as e:
                logger.error(f"Failed embedding for paragraph {paragraph_id}: {e}")
                continue
        
        return vectors
    
    async def save_article_embedding(
        self,
        session: AsyncSession,
        article_id: int,
        paragraph_vectors: List[List[float]]
    ) -> None:
        """
        Compute and save article-level embedding from paragraph vectors.
        
        Args:
            session: Database session
            article_id: Article ID
            paragraph_vectors: List of paragraph embedding vectors
        """
        if not paragraph_vectors:
            logger.warning(f"No vectors to average for article {article_id}")
            return
        
        # Clear existing
        await session.execute(
            text("DELETE FROM article_embedding_1024 WHERE article_id = :article_id"),
            {"article_id": article_id}
        )
        
        avg_vector = np.mean(paragraph_vectors, axis=0).tolist()
        
        embedding = ArticleEmbedding1024(
            article_id=article_id,
            embedding=avg_vector
        )
        session.add(embedding)
        
        logger.info(f"Saved article embedding for {article_id}")
    
    async def generate_and_save_all(
        self,
        session: AsyncSession,
        article_id: int,
        paragraph_data: List[Tuple[int, str]]
    ) -> None:
        """
        Generate and save both paragraph and article embeddings.
        
        Args:
            session: Database session
            article_id: Article ID
            paragraph_data: List of (paragraph_id, content) tuples
        """
        logger.info(f"Generating embeddings for {len(paragraph_data)} paragraphs")
        
        vectors = await self.save_paragraph_embeddings(session, article_id, paragraph_data)
        
        if vectors:
            await self.save_article_embedding(session, article_id, vectors)
            await session.commit()
            logger.info(f"Successfully saved all embeddings for article {article_id}")
        else:
            logger.warning(f"No embeddings generated for article {article_id}")


def get_embedding_provider() -> IEmbeddingProvider:
    """Factory for getting embedding provider based on configuration."""
    from ....core.config import get_settings
    settings = get_settings()
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "ollama":
        from .ollama_embedding_provider import OllamaEmbeddingProvider
        return OllamaEmbeddingProvider()
    elif provider == "llamacpp":
        from .llamacpp_embedding_provider import LlamaCppEmbeddingProvider
        return LlamaCppEmbeddingProvider()
    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            f"Supported options are: ollama, llamacpp"
        )


# Singleton instance
_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create EmbeddingService singleton."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService(provider=get_embedding_provider())
    return _embedding_service_instance
