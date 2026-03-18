"""Type-safe HTTP client for internal module communication."""
from typing import TypeVar, Optional, Dict, Any, Type
import httpx
from enum import Enum
from pydantic import BaseModel
from ..core.base_dtos import ResponseDto
from ..core.config import get_settings


class HttpMethod(str, Enum):
    """HTTP methods enum."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


T = TypeVar("T", bound=BaseModel)


class ModulesHttpClient:
    """
    Type-safe HTTP client for internal module-to-module communication.
    
    This client:
    - Always uses SELF_URL as the base URL
    - Returns responses wrapped in ResponseDto[T]
    - Provides type safety for response data
    """
    
    def __init__(self, timeout: float = 30.0):
        """
        Initialize modules HTTP client.
        
        Args:
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.settings = get_settings()
        self.base_url = self.settings.SELF_URL
        self.timeout = timeout
        
    async def request(
        self,
        method: HttpMethod,
        url: str,
        response_type: Type[T],
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseDto[T]:
        """
        Make a type-safe HTTP request to an internal module.
        
        Args:
            method: HTTP method to use
            url: URL path (relative to base URL)
            response_type: Pydantic model type for the response data
            params: Query parameters
            json: JSON body data
            data: Form data
            headers: Request headers
            timeout: Override default timeout
            
        Returns:
            ResponseDto[T] with typed data
            
        Raises:
            httpx.HTTPError: On request failure
        """
        # Build full URL
        full_url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
        
        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self.timeout
        
        # Prepare headers
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            response = await client.request(
                method=method.value,
                url=full_url,
                params=params,
                json=json,
                data=data,
                headers=request_headers,
            )
            response.raise_for_status()
            
            # Parse response as ResponseDto[T]
            response_data = response.json()
            
            # If the response has a 'data' field, parse it with the response_type
            if response_data.get("data") is not None:
                typed_data = response_type(**response_data["data"])
                return ResponseDto[response_type](
                    data=typed_data,
                    status_code=response_data.get("status_code", 0),
                    error_description=response_data.get("error_description"),
                )
            else:
                return ResponseDto[response_type](
                    data=None,
                    status_code=response_data.get("status_code", 0),
                    error_description=response_data.get("error_description"),
                )
    
    async def get(
        self,
        url: str,
        response_type: Type[T],
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseDto[T]:
        """Make a type-safe GET request."""
        return await self.request(
            HttpMethod.GET,
            url,
            response_type,
            params=params,
            headers=headers,
            timeout=timeout,
        )
    
    async def post(
        self,
        url: str,
        response_type: Type[T],
        *,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseDto[T]:
        """Make a type-safe POST request."""
        return await self.request(
            HttpMethod.POST,
            url,
            response_type,
            json=json,
            data=data,
            params=params,
            headers=headers,
            timeout=timeout,
        )
    
    async def put(
        self,
        url: str,
        response_type: Type[T],
        *,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseDto[T]:
        """Make a type-safe PUT request."""
        return await self.request(
            HttpMethod.PUT,
            url,
            response_type,
            json=json,
            data=data,
            params=params,
            headers=headers,
            timeout=timeout,
        )
    
    async def patch(
        self,
        url: str,
        response_type: Type[T],
        *,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseDto[T]:
        """Make a type-safe PATCH request."""
        return await self.request(
            HttpMethod.PATCH,
            url,
            response_type,
            json=json,
            data=data,
            params=params,
            headers=headers,
            timeout=timeout,
        )
    
    async def delete(
        self,
        url: str,
        response_type: Type[T],
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ResponseDto[T]:
        """Make a type-safe DELETE request."""
        return await self.request(
            HttpMethod.DELETE,
            url,
            response_type,
            params=params,
            headers=headers,
            timeout=timeout,
        )
