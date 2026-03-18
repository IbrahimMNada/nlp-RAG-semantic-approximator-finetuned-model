"""SEO generation module dependencies."""
from typing import Annotated
from functools import lru_cache
from fastapi import Depends
from .services import SeoService
from .services.dataset_service import DatasetService


@lru_cache(maxsize=1)
def get_seo_service() -> SeoService:
    """
    Dependency for getting SEO service singleton instance.
    Uses lru_cache to ensure only one instance is created for the entire application.
    
    Returns:
        SeoService singleton instance
    """
    return SeoService()


@lru_cache(maxsize=1)
def get_dataset_service() -> DatasetService:
    """
    Dependency for getting Dataset service singleton instance.
    Uses lru_cache to ensure only one instance is created for the entire application.
    
    Returns:
        DatasetService singleton instance
    """
    return DatasetService()


# Type alias for dependency injection
SeoServiceDep = Annotated[SeoService, Depends(get_seo_service)]
DatasetServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
