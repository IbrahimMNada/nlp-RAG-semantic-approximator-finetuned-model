"""
Startup service that ensures all required Ollama models are available.
Pulls missing models automatically when AUTO_PULL_MODELS is enabled.
"""
import logging
from typing import List

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)


def _get_required_models() -> List[str]:
    """Collect all Ollama model names that the current config requires."""
    settings = get_settings()
    models = []

    # Embedding model — always needed
    models.append(settings.OLLAMA_MODEL_NAME)

    # Reranker model — only if reranker is enabled
    if settings.RERANKER_ENABLED:
        models.append(settings.RERANKER_MODEL)

    # Ollama LLM model — only if Ollama is the active LLM provider
    if settings.LLM_PROVIDER == "ollama":
        models.append(settings.OLLAMA_LLM_MODEL)

    return list(set(models))


async def _get_installed_models(base_url: str) -> List[str]:
    """Fetch the list of models already pulled in Ollama."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{base_url}/api/tags")
        response.raise_for_status()
        data = response.json()

    return [
        m.get("name", m.get("model", ""))
        for m in data.get("models", [])
    ]


async def _pull_model(base_url: str, model_name: str) -> None:
    """Pull a single model from Ollama."""
    logger.info(f"Pulling model '{model_name}' — this may take a while...")
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{base_url}/api/pull",
            json={"name": model_name, "stream": False},
        )
        response.raise_for_status()
    logger.info(f"Model '{model_name}' pulled successfully")


async def ensure_models() -> None:
    """
    Check all required Ollama models and pull any that are missing.
    Skipped entirely when AUTO_PULL_MODELS is False.
    """
    settings = get_settings()
    if not settings.AUTO_PULL_MODELS:
        return

    required = _get_required_models()
    if not required:
        return

    base_url = settings.OLLAMA_URL

    try:
        installed = await _get_installed_models(base_url)
    except Exception as e:
        logger.warning(f"Could not connect to Ollama at {base_url}: {e}")
        return

    for model in required:
        already_present = any(model in name for name in installed)
        if already_present:
            logger.info(f"Model '{model}' already available")
        else:
            try:
                await _pull_model(base_url, model)
            except Exception as e:
                logger.error(f"Failed to pull model '{model}': {e}")
