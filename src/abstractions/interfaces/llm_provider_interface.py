"""Abstract interface for LLM (Language Model) providers."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class ILLMProvider(ABC):
    """
    Interface for Language Model providers.
    
    This abstraction allows different LLM providers (OpenAI, Claude, DeepSeek, etc.)
    to be used interchangeably in the application.
    """
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            model: Model to use (optional, can use provider default)
            
        Returns:
            Dict containing the response with structure:
            {
                "choices": [
                    {
                        "message": {
                            "content": str
                        }
                    }
                ],
                "usage": {
                    "total_tokens": int
                }
            }
            
        Raises:
            Exception: On API request failure
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def build_prompt(
        self,
        context: str,
        question: str,
        has_context: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Build provider-specific prompt from context and question.
        
        Each provider can customize how they format the prompt based on their
        model's strengths and prompt engineering best practices.
        
        Args:
            context: Retrieved context from knowledge base
            question: User's question
            has_context: Whether context was retrieved
            
        Returns:
            List of message dicts formatted for the provider
        """
        pass
    
    @abstractmethod
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate the cost of the API call based on token usage.
        
        Args:
            input_tokens: Number of tokens in the input/prompt
            output_tokens: Number of tokens in the output/completion
            
        Returns:
            Cost in USD
        """
        pass
