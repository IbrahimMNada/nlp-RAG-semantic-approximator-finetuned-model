"""
Example Claude consumer implementation (requires anthropic package).

To use Claude:
1. Install: pip install anthropic
2. Add to .env: CLAUDE_API_KEY=your-key-here
3. Update dependencies.py: return ClaudeConsumer() instead of ChatGPTConsumer()
"""
from typing import List, Dict, Any, Optional
# from anthropic import AsyncAnthropic  # Uncomment when anthropic is installed
from ....core.config import get_settings
from ....abstractions.interfaces.llm_provider_interface import ILLMProvider


class ClaudeConsumer(ILLMProvider):
    """
    Consumer for interacting with Anthropic's Claude API.
    Implements the ILLMProvider interface.
    
    NOTE: This is a template/example implementation.
    Uncomment the imports and implementation when ready to use.
    """
    
    def __init__(self):
        """Initialize Claude consumer with settings."""
        self.settings = get_settings()
        # Uncomment when anthropic package is installed:
        # self.client = AsyncAnthropic(
        #     api_key=self.settings.CLAUDE_API_KEY,
        #     timeout=60.0,
        # )
        raise NotImplementedError(
            "Claude consumer requires 'anthropic' package. "
            "Install with: pip install anthropic"
        )
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a chat completion using Claude API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            model: Model to use (defaults to claude-3-sonnet)
            
        Returns:
            Dict containing the API response in OpenAI-compatible format
        """
        # Example implementation (uncomment when ready):
        # response = await self.client.messages.create(
        #     model=model or "claude-3-sonnet-20240229",
        #     messages=messages,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        # )
        # 
        # # Convert to OpenAI-compatible format
        # return {
        #     "choices": [
        #         {
        #             "message": {
        #                 "content": response.content[0].text
        #             }
        #         }
        #     ],
        #     "usage": {
        #         "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        #     }
        # }
        raise NotImplementedError("Claude implementation is not yet enabled")
    
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
        Build Claude-optimized prompt.
        
        Claude performs well with XML-style tags and clear structure.
        """
        messages = []
        
        if has_context:
            # Claude works well with XML tags for structure
            system_prompt = (
                "You are a helpful AI assistant. Use the information provided in the <context> tags to answer the user's question. "
                "Be thorough and cite specific information from the context when relevant.\n\n"
                f"<context>\n{context}\n</context>\n\n"
                "If the context doesn't contain the information needed to answer the question, say so clearly."
            )
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant. Provide clear, accurate answers to user questions."
            })
        
        messages.append({"role": "user", "content": question})
        return messages
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for Anthropic Claude API.
        
        Pricing as of 2024 (update as needed):
        - Claude 3 Sonnet: $3 / 1M input tokens, $15 / 1M output tokens
        - Claude 3 Opus: $15 / 1M input tokens, $75 / 1M output tokens
        - Claude 3 Haiku: $0.25 / 1M input tokens, $1.25 / 1M output tokens
        """
        model = self.settings.CLAUDE_MODEL.lower()
        
        # Pricing per million tokens
        if "opus" in model:
            input_cost = 15.0 / 1_000_000
            output_cost = 75.0 / 1_000_000
        elif "haiku" in model:
            input_cost = 0.25 / 1_000_000
            output_cost = 1.25 / 1_000_000
        else:
            # Sonnet (default)
            input_cost = 3.0 / 1_000_000
            output_cost = 15.0 / 1_000_000
        
        total_cost = (input_tokens * input_cost) + (output_tokens * output_cost)
        return round(total_cost, 6)
