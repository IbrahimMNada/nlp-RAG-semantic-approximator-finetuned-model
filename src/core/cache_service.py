import json
from typing import Optional, List, Any
import logging

logger = logging.getLogger(__name__)

# Optional Redis import - fail silently if not available
try:
    from redis import asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis package not installed. Caching will be disabled.")
    REDIS_AVAILABLE = False
    aioredis = Any  # type: ignore


class RedisCacheService:
    """
    Redis cache service for similarity search results.
    Manages caching with URL and threshold as key components.
    Gracefully degrades if Redis is not available.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """
        Initialize Redis cache service.
        
        Args:
            redis_url: Redis connection URL
        """
        self._redis: Optional[Any] = None
        self.redis_url = redis_url
        self._connected = False
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - caching disabled")
    
    async def connect(self):
        """Initialize Redis connection. Fails silently if Redis unavailable."""
        if not REDIS_AVAILABLE:
            return
        
        try:
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10
            )
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis connection failed, caching disabled: {str(e)}")
            self._connected = False
            self._redis = None
    
    async def disconnect(self):
        """Close Redis connection."""
        if self._redis:
            try:
                await self._redis.aclose()
                self._connected = False
                logger.info("Redis cache disconnected")
            except Exception:
                pass
    
    def _generate_cache_key(self, url: str, threshold: float = 0.0) -> str:
        """
        Generate cache key from URL and threshold.
        
        Args:
            url: Article URL
            threshold: Similarity threshold
            
        Returns:
            Cache key string
        """
        return f"similarity:{url}:{threshold}"
    
    async def get_cached_results(
        self, 
        url: str, 
        requested_limit: int,
        threshold: float = 0.0
    ) -> Optional[List[dict]]:
        """
        Get cached similarity results.
        Returns cached results only if cached limit >= requested limit.
        
        Args:
            url: Article URL
            requested_limit: Number of results requested
            threshold: Similarity threshold
            
        Returns:
            List of similar articles or None if not cached or limit insufficient
        """
        if not self._connected or not self._redis:
            return None
        
        try:
            key = self._generate_cache_key(url, threshold)
            cached = await self._redis.get(key)
            
            if cached:
                data = json.loads(cached)
                cached_limit = data.get("limit", 0)
                cached_results = data.get("results", [])
                
                # If cached limit >= requested limit, return subset
                if cached_limit >= requested_limit:
                    logger.info(f"Cache HIT for {url} (cached: {cached_limit}, requested: {requested_limit})")
                    # Return only the requested number of results
                    return cached_results[:requested_limit]
                else:
                    logger.info(f"Cache MISS - limit too small (cached: {cached_limit}, requested: {requested_limit})")
                    return None
            
            logger.info(f"Cache MISS for {url}")
            return None
            
        except Exception as e:
            logger.debug(f"Cache get error (silent fail): {str(e)}")
            return None
    
    async def set_cached_results(
        self,
        url: str,
        results: List[dict],
        limit: int,
        threshold: float = 0.0,
        ttl: int = 3600
    ):
        """
        Cache similarity search results.
        Overwrites existing cache with new limit if new limit is larger.
        
        Args:
            url: Article URL
            results: List of similar articles
            limit: Number of results (limit used for the search)
            threshold: Similarity threshold
            ttl: Time to live in seconds (default: 1 hour)
        """
        if not self._connected or not self._redis:
            return
        
        try:
            key = self._generate_cache_key(url, threshold)
            
            # Store both results and the limit
            cache_data = {
                "limit": limit,
                "results": results,
                "threshold": threshold
            }
            
            await self._redis.setex(
                key,
                ttl,
                json.dumps(cache_data)
            )
            
            logger.info(f"Cached {len(results)} results for {url} with limit={limit} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.debug(f"Cache set error (silent fail): {str(e)}")
    
    async def clear_all_similarity_cache(self):
        """
        Clear all similarity search cache entries.
        Used when rebuilding the index.
        """
        if not self._connected or not self._redis:
            return
        
        try:
            # Find all keys matching the similarity pattern
            pattern = "similarity:*"
            deleted_count = 0
            
            # Use SCAN to iterate through keys safely
            async for key in self._redis.scan_iter(match=pattern, count=100):
                await self._redis.delete(key)
                deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} similarity cache entries")
            return deleted_count
            
        except Exception as e:
            logger.debug(f"Cache clear error (silent fail): {str(e)}")
            return 0
    
    async def invalidate_url(self, url: str):
        """
        Invalidate all cached entries for a specific URL.
        
        Args:
            url: Article URL to invalidate
        """
        if not self._connected or not self._redis:
            return
        
        try:
            # Find all keys for this URL with any threshold
            pattern = f"similarity:{url}:*"
            deleted_count = 0
            
            async for key in self._redis.scan_iter(match=pattern, count=100):
                await self._redis.delete(key)
                deleted_count += 1
            
            logger.info(f"Invalidated {deleted_count} cache entries for {url}")
            
        except Exception as e:
            logger.debug(f"Cache invalidation error (silent fail): {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if connected and responsive, False otherwise
        """
        if not self._redis:
            return False
        try:
            return await self._redis.ping()
        except Exception:
            return False


# Global cache instance (will be configured from settings on startup)
def _get_redis_url() -> str:
    """Get Redis URL from settings, with fallback to default."""
    try:
        from .config import get_settings
        settings = get_settings()
        return settings.REDIS_URL
    except Exception:
        return "redis://localhost:6379/0"

cache_service = RedisCacheService(redis_url=_get_redis_url())
