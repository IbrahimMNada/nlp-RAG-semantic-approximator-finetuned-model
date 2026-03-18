"""Ollama consumer for local LLM inference."""
from typing import List, Dict, Any, Optional
import httpx
from ....core.config import get_settings
from ....abstractions.interfaces.llm_provider_interface import ILLMProvider


class OllamaConsumer(ILLMProvider):
    """
    Consumer for interacting with Ollama API.
    Ollama runs locally and supports various open-source models.
    Implements the ILLMProvider interface.
    """
    
    def __init__(self):
        """Initialize Ollama consumer with settings."""
        self.settings = get_settings()
        self.base_url = self.settings.OLLAMA_LLM_URL
        self.default_model = self.settings.OLLAMA_LLM_MODEL
        self.timeout = 120.0  # Longer timeout for local inference
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a chat completion using Ollama API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            model: Model to use (defaults to config setting)
            
        Returns:
            Dict containing the API response in OpenAI-compatible format
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model or self.default_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                }
            )
            response.raise_for_status()
            ollama_response = response.json()
            
            # Convert Ollama response to OpenAI-compatible format
            return {
                "choices": [
                    {
                        "message": {
                            "content": ollama_response.get("message", {}).get("content", "")
                        }
                    }
                ],
                "usage": {
                    "total_tokens": (
                        ollama_response.get("prompt_eval_count", 0) + 
                        ollama_response.get("eval_count", 0)
                    )
                }
            }
    
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
        Build Ollama-optimized prompt.
        
        Ollama models vary, but generally work well with clear, direct instructions.
        Optimized for local models which may have smaller context windows.
        """
        messages = []
        
        if has_context:
            # Keep it concise for local models with smaller context
            system_prompt = (
                "Use the following information to answer the question. If the information doesn't help, say so.\n\n"
                f"{context}"
            )
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant."
            })
        
        messages.append({"role": "user", "content": question})
        return messages
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for Ollama.
        
        Ollama is free (runs locally), so cost is always $0.
        You might want to track compute costs separately if desired.
        """
        return 0.0  # Free - runs locally
