"""
Startup service that ensures all required Ollama models are available.
Pulls missing models automatically when AUTO_PULL_MODELS is enabled.
"""
import logging
import sys
import time
from typing import List

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

# ANSI colors for terminal output
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_CHECK = "✔"
_CROSS = "✘"
_ARROW = "→"
_DOWNLOAD = "⬇"


def _log_step(icon: str, color: str, message: str) -> None:
    """Print a formatted startup step to stderr (bypasses log formatting)."""
    sys.stderr.write(f"  {color}{icon}{_RESET} {message}\n")
    sys.stderr.flush()


def _get_required_models() -> List[str]:
    """Collect all Ollama model names that the current config requires."""
    settings = get_settings()
    models = []

    # Embedding model — always needed
    models.append(settings.OLLAMA_MODEL_NAME)

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


def _format_bytes(n: int) -> str:
    """Format byte count to human readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _render_progress_bar(completed: int, total: int, width: int = 30) -> str:
    """Render a text progress bar."""
    if total == 0:
        return ""
    ratio = min(completed / total, 1.0)
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    pct = ratio * 100
    return f"[{bar}] {pct:5.1f}% {_format_bytes(completed)}/{_format_bytes(total)}"


async def _pull_model(base_url: str, model_name: str) -> None:
    """Pull a single model from Ollama with streaming progress."""
    logger.info(f"Pulling model '{model_name}'...")
    _log_step(_DOWNLOAD, _CYAN, f"Pulling {_BOLD}{model_name}{_RESET} ...")

    start_time = time.time()

    async with httpx.AsyncClient(timeout=600.0) as client:
        async with client.stream(
            "POST",
            f"{base_url}/api/pull",
            json={"name": model_name, "stream": True},
        ) as response:
            response.raise_for_status()

            last_status = ""
            async for line in response.aiter_lines():
                if not line:
                    continue
                import json as _json
                try:
                    chunk = _json.loads(line)
                except Exception:
                    continue

                status = chunk.get("status", "")
                total = chunk.get("total", 0)
                completed = chunk.get("completed", 0)

                if total and completed:
                    bar = _render_progress_bar(completed, total)
                    sys.stderr.write(f"\r  {_CYAN}{_DOWNLOAD}{_RESET} {status}: {bar}")
                    sys.stderr.flush()
                elif status and status != last_status:
                    sys.stderr.write(f"\r  {_CYAN}{_ARROW}{_RESET} {status:<60}\n")
                    sys.stderr.flush()

                last_status = status

    elapsed = time.time() - start_time
    sys.stderr.write("\n")
    _log_step(_CHECK, _GREEN, f"{_BOLD}{model_name}{_RESET} pulled in {elapsed:.1f}s")
    logger.info(f"Model '{model_name}' pulled successfully in {elapsed:.1f}s")


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

    sys.stderr.write(f"\n{_BOLD}{'─' * 50}{_RESET}\n")
    sys.stderr.write(f"  {_BOLD}Ollama Model Check{_RESET}  ({base_url})\n")
    sys.stderr.write(f"{_BOLD}{'─' * 50}{_RESET}\n")
    sys.stderr.flush()

    try:
        installed = await _get_installed_models(base_url)
        _log_step(_CHECK, _GREEN, f"Connected to Ollama ({len(installed)} models installed)")
    except Exception as e:
        _log_step(_CROSS, _RED, f"Could not connect to Ollama at {base_url}: {e}")
        logger.warning(f"Could not connect to Ollama at {base_url}: {e}")
        sys.stderr.write(f"{_BOLD}{'─' * 50}{_RESET}\n\n")
        return

    _log_step(_ARROW, _CYAN, f"Required models: {', '.join(required)}")

    pulled = 0
    skipped = 0
    failed = 0

    for i, model in enumerate(required, 1):
        already_present = any(model in name for name in installed)
        if already_present:
            _log_step(_CHECK, _GREEN, f"[{i}/{len(required)}] {model} — already available")
            logger.info(f"Model '{model}' already available")
            skipped += 1
        else:
            try:
                await _pull_model(base_url, model)
                pulled += 1
            except Exception as e:
                _log_step(_CROSS, _RED, f"[{i}/{len(required)}] {model} — failed: {e}")
                logger.error(f"Failed to pull model '{model}': {e}")
                failed += 1

    # Summary
    summary_parts = []
    if skipped:
        summary_parts.append(f"{_GREEN}{skipped} ready{_RESET}")
    if pulled:
        summary_parts.append(f"{_CYAN}{pulled} pulled{_RESET}")
    if failed:
        summary_parts.append(f"{_RED}{failed} failed{_RESET}")

    sys.stderr.write(f"  {_BOLD}{_ARROW}{_RESET} Summary: {', '.join(summary_parts)}\n")
    sys.stderr.write(f"{_BOLD}{'─' * 50}{_RESET}\n\n")
    sys.stderr.flush()
