"""
Service for generating and managing vector embeddings.
"""
import logging
from typing import List, Tuple, Optional

import numpy as np
from ollama import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.config import get_settings
from ....shared.text_utils import normalize_arabic
from ..entities import ParagraphEmbedding1024, ArticleEmbedding1024

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles embedding generation and storage."""
    
    def __init__(self):
        self._client: Optional[AsyncClient] = None
        self._model_pulled: bool = False
    
    def _get_client(self) -> AsyncClient:
        """Lazy initialization of Ollama client."""
        if self._client is None:
            settings = get_settings()
            self._client = AsyncClient(host=settings.OLLAMA_URL)
        return self._client

    async def _ensure_model(self) -> None:
        """Pull the embedding model if not already available."""
        if self._model_pulled:
            return
        settings = get_settings()
        client = self._get_client()
        try:
            models = await client.list()
            model_names = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
            if not any(settings.OLLAMA_MODEL_NAME in name for name in model_names):
                logger.info(f"Pulling model '{settings.OLLAMA_MODEL_NAME}'...")
                await client.pull(settings.OLLAMA_MODEL_NAME)
                logger.info(f"Model '{settings.OLLAMA_MODEL_NAME}' pulled successfully")
            self._model_pulled = True
        except Exception as e:
            logger.warning(f"Could not verify/pull model: {e}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed (will be normalized)
            
        Returns:
            Embedding vector
        """
        settings = get_settings()
        client = self._get_client()
        await self._ensure_model()
        normalized = normalize_arabic(text)
        response = await client.embed(
            model=settings.OLLAMA_MODEL_NAME,
            input=normalized
        )
        return response["embeddings"][0]
    
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


# Singleton instance
_embedding_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create EmbeddingService singleton."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance
