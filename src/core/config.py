from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "NLP Similarity Approximator"
    APP_VERSION: str = "0.1.0"
    DATABASE_URL: str = ""
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "nomic-embed-text"
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600  # 1 hour
    
    # RabbitMQ settings
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
    RABBITMQ_QUEUE_NAME: str = "embeddings_queue"
    QUEUE_ENABLED: bool = False  # Set to True to use queue-based processing
    
    # Content filtering - paragraphs containing these strings will be skipped
    SKIP_PARAGRAPHS_CONTAINING: List[str] = ["تم الإرسال بنجاح، شكراً لك!"]
    
    # OpenAI settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    
    # LLM Provider settings
    LLM_PROVIDER: str = "chatgpt"  # Options: chatgpt, claude, deepseek, ollama
    CLAUDE_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-3-sonnet-20240229"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    OLLAMA_LLM_URL: str = "http://localhost:11434"  # Ollama for chat/LLM
    OLLAMA_LLM_MODEL: str = "llama3.2"  # Model for chat completions
    
    # Seq logging settings
    SEQ_SERVER_URL: str = ""  # e.g., "http://localhost:5341"
    SEQ_API_KEY: str = ""
    SEQ_ENABLED: bool = False
    
    HF_TOKEN: str = ""  # Hugging Face API token

    # API Key authentication
    API_KEY_ENABLED: bool = False  # Set to True to require API key
    API_KEY: str = ""  # API key value when enabled

    # CORS settings
    CORS_ALLOWED_ORIGINS: List[str] = ["*"]  # Restrict in production

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance - parsed once at startup."""
    return Settings()