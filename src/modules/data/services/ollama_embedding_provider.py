"""Ollama embedding provider."""
import logging
from typing import List, Optional

from ollama import AsyncClient

from ....core.config import get_settings
from ....abstractions.interfaces.embedding_provider_interface import IEmbeddingProvider
from ....shared.text_utils import normalize_arabic

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(IEmbeddingProvider):
    """Generates embeddings using the Ollama API."""

    def __init__(self):
        self._client: Optional[AsyncClient] = None
        self._model_pulled: bool = False

    def _get_client(self) -> AsyncClient:
        if self._client is None:
            settings = get_settings()
            self._client = AsyncClient(host=settings.OLLAMA_URL)
        return self._client

    async def _ensure_model(self) -> None:
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
        settings = get_settings()
        client = self._get_client()
        await self._ensure_model()
        normalized = normalize_arabic(text)
        response = await client.embed(
            model=settings.OLLAMA_MODEL_NAME,
            input=normalized,
        )
        return response["embeddings"][0]
