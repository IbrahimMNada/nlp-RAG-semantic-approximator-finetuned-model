"""ChatGPT consumer for communicating with OpenAI API."""
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from ....core.config import get_settings
from ....abstractions.interfaces.llm_provider_interface import ILLMProvider


class ChatGPTConsumer(ILLMProvider):
    """
    Consumer for interacting with OpenAI's ChatGPT API.
    Uses the official OpenAI Python SDK.
    Implements the ILLMProvider interface.
    """
    
    def __init__(self):
        """Initialize ChatGPT consumer with settings."""
        self.settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            base_url=self.settings.OPENAI_API_BASE,
            timeout=60.0,
        )
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a chat completion using OpenAI API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            model: Model to use (defaults to config setting)
            
        Returns:
            Dict containing the API response
            
        Raises:
            openai.APIError: On API request failure
        """
        response = await self.client.chat.completions.create(
            model=model or self.settings.OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Convert to dict for backward compatibility
        return response.model_dump()
    
    async def extract_message_content(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """
        Get chat completion and extract just the message content.
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            String content of the assistant's response
        """
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
        Build ChatGPT-optimized prompt.
        
        ChatGPT works well with clear system instructions and structured context.
        """
        messages = []
        
        if has_context:
            # ChatGPT benefits from explicit instructions
            system_prompt = (
                "You are a knowledgeable assistant. Answer the user's question based on the provided context. "
                "If the context contains relevant information, use it to provide a detailed and accurate answer. "
                "If the context doesn't contain relevant information, clearly state that and provide a general answer if possible.\n\n"
                f"Context:\n{context}"
            )
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's question to the best of your ability."
            })
        
        messages.append({"role": "user", "content": question})
        return messages
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for OpenAI API.
        
        Pricing as of 2024 (update as needed):
        - GPT-3.5-turbo: $0.50 / 1M input tokens, $1.50 / 1M output tokens
        - GPT-4: $30 / 1M input tokens, $60 / 1M output tokens
        - GPT-4-turbo: $10 / 1M input tokens, $30 / 1M output tokens
        """
        model = self.settings.OPENAI_MODEL.lower()
        
        # Pricing per million tokens
        if "gpt-4" in model and "turbo" in model:
            # GPT-4 Turbo
            input_cost = 10.0 / 1_000_000
            output_cost = 30.0 / 1_000_000
        elif "gpt-4" in model:
            # GPT-4
            input_cost = 30.0 / 1_000_000
            output_cost = 60.0 / 1_000_000
        else:
            # GPT-3.5-turbo (default)
            input_cost = 0.50 / 1_000_000
            output_cost = 1.50 / 1_000_000
        
        total_cost = (input_tokens * input_cost) + (output_tokens * output_cost)
        return round(total_cost, 6)
