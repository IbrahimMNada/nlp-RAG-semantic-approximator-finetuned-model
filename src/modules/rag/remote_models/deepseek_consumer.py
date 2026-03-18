"""
Example DeepSeek consumer implementation.

To use DeepSeek:
1. Add to .env: DEEPSEEK_API_KEY=your-key-here
2. Update dependencies.py: return DeepSeekConsumer() instead of ChatGPTConsumer()
"""
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from ....core.config import get_settings
from ....abstractions.interfaces.llm_provider_interface import ILLMProvider


class DeepSeekConsumer(ILLMProvider):
    """
    Consumer for interacting with DeepSeek API.
    DeepSeek is OpenAI-compatible, so we can use the OpenAI SDK.
    Implements the ILLMProvider interface.
    
    NOTE: This is a template/example implementation.
    Update the configuration when ready to use.
    """
    
    def __init__(self):
        """Initialize DeepSeek consumer with settings."""
        self.settings = get_settings()
        # DeepSeek is OpenAI-compatible, just different base URL and key
        self.client = AsyncOpenAI(
            api_key=self.settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",  # DeepSeek API endpoint
            timeout=60.0,
        )
        self.default_model = self.settings.DEEPSEEK_MODEL
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a chat completion using DeepSeek API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            model: Model to use (defaults to deepseek-chat)
            
        Returns:
            Dict containing the API response
        """
        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Convert to dict
        return response.model_dump()
    
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
        Build DeepSeek-optimized prompt.
        
        DeepSeek is similar to OpenAI format but benefits from concise instructions.
        """
        messages = []
        
        if has_context:
            system_prompt = (
                "Answer the question using the context provided below. Be precise and cite the context when applicable.\n\n"
                f"Context:\n{context}"
            )
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant. Answer questions accurately and concisely."
            })
        
        messages.append({"role": "user", "content": question})
        return messages
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for DeepSeek API.
        
        DeepSeek pricing (as of 2024):
        - DeepSeek-Chat: $0.14 / 1M input tokens, $0.28 / 1M output tokens
        Very cost-effective compared to other providers.
        """
        # DeepSeek pricing per million tokens
        input_cost = 0.14 / 1_000_000
        output_cost = 0.28 / 1_000_000
        
        total_cost = (input_tokens * input_cost) + (output_tokens * output_cost)
        return round(total_cost, 6)
