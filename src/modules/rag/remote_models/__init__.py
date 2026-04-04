"""LLM provider consumers for RAG module."""
from .chatgpt_consumer import ChatGPTConsumer
# from .claude_consumer import ClaudeConsumer  # Uncomment when ready to use
# from .deepseek_consumer import DeepSeekConsumer  # Uncomment when ready to use
# from .ollama_consumer import OllamaConsumer  # Uncomment when ready to use
# from .llamacpp_consumer import LlamaCppConsumer  # Uncomment when ready to use

__all__ = ["ChatGPTConsumer"]
