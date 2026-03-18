"""
Shared security dependencies: API key auth and rate limiting.
"""
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from .config import get_settings

# Rate limiter (keyed by client IP)
limiter = Limiter(key_func=get_remote_address)

# API key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key if authentication is enabled."""
    settings = get_settings()
    if not settings.API_KEY_ENABLED:
        return None
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
