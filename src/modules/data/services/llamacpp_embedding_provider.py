"""llama.cpp server embedding provider."""
import logging
from typing import List

import httpx

from ....core.config import get_settings
from ....abstractions.interfaces.embedding_provider_interface import IEmbeddingProvider
from ....shared.text_utils import normalize_arabic

logger = logging.getLogger(__name__)


class LlamaCppEmbeddingProvider(IEmbeddingProvider):
    """Generates embeddings using llama.cpp's OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.LLAMA_CPP_EMBEDDING_URL.rstrip("/")
        self.model = settings.LLAMA_CPP_EMBEDDING_MODEL
        self.timeout = 60.0

    async def generate_embedding(self, text: str) -> List[float]:
        normalized = normalize_arabic(text)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "model": self.model,
                    "input": normalized,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
