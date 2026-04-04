"""llama.cpp server consumer for local LLM inference."""
from typing import List, Dict, Any, Optional
import httpx
from ....core.config import get_settings
from ....abstractions.interfaces.llm_provider_interface import ILLMProvider


class LlamaCppConsumer(ILLMProvider):
    """
    Consumer for interacting with llama.cpp's OpenAI-compatible server.
    Expects llama-server running with --port matching LLAMA_CPP_URL.
    Implements the ILLMProvider interface.
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.LLAMA_CPP_URL.rstrip("/")
        self.default_model = self.settings.LLAMA_CPP_MODEL
        self.timeout = 120.0

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a chat completion using llama.cpp's OpenAI-compatible endpoint.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            model: Model identifier (optional, llama-server typically serves one model)

        Returns:
            Dict containing the API response in OpenAI-compatible format
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model or self.default_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            return response.json()

    async def extract_message_content(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Get chat completion and extract just the message content."""
        response = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]

    def build_prompt(
        self,
        context: str,
        question: str,
        has_context: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Build prompt for llama.cpp served models.

        Works well with direct, concise instructions similar to Ollama
        since llama.cpp typically serves the same open-source models.
        """
        messages = []

        if has_context:
            system_prompt = (
                "Use the following information to answer the question. "
                "If the information doesn't help, say so.\n\n"
                f"{context}"
            )
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant.",
            })

        messages.append({"role": "user", "content": question})
        return messages

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for llama.cpp.

        llama.cpp runs locally, so cost is always $0.
        """
        return 0.0
