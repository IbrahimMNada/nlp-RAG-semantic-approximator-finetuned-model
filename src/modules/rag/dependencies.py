"""RAG module dependencies."""
from typing import Annotated
from fastapi import Depends
from .services import RagService
from .remote_models import ChatGPTConsumer
from ...abstractions.interfaces.llm_provider_interface import ILLMProvider
from ...core.config import get_settings


def get_llm_provider() -> ILLMProvider:
    """
    Factory for getting LLM provider instance based on configuration.
    
    Reads LLM_PROVIDER from environment variables to determine which provider to use.
    Supported providers: chatgpt, claude, deepseek, ollama
    
    Returns:
        ILLMProvider instance
        
    Raises:
        ValueError: If an unsupported provider is specified
    """
    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "chatgpt":
        return ChatGPTConsumer()
    
    elif provider == "claude":
        from .remote_models.claude_consumer import ClaudeConsumer
        return ClaudeConsumer()
    
    elif provider == "deepseek":
        from .remote_models.deepseek_consumer import DeepSeekConsumer
        return DeepSeekConsumer()
    
    elif provider == "ollama":
        from .remote_models.ollama_consumer import OllamaConsumer
        return OllamaConsumer()
    
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported options are: chatgpt, claude, deepseek, ollama"
        )


def get_rag_service(
    llm_provider: Annotated[ILLMProvider, Depends(get_llm_provider)]
) -> RagService:
    """
    Dependency for getting RAG service instance.
    
    Args:
        llm_provider: LLM provider implementation
    
    Returns:
        RagService instance
    """
    return RagService(llm_provider=llm_provider)


# Type aliases for dependency injection
RagServiceDep = Annotated[RagService, Depends(get_rag_service)]
