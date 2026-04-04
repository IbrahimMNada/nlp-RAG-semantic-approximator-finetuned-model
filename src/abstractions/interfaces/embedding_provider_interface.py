"""Abstract interface for embedding providers."""
from abc import ABC, abstractmethod
from typing import List


class IEmbeddingProvider(ABC):
    """
    Interface for embedding providers.

    Allows different embedding backends (Ollama, llama.cpp, etc.)
    to be used interchangeably for vector generation.
    """

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single text.

        Args:
            text: The input text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        pass
