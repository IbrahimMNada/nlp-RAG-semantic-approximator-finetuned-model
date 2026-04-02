"""
Data module dependencies - all injectable services for data module.
"""
from typing import Annotated
from functools import lru_cache

from fastapi import Depends

from ...abstractions.interfaces.web_scraper_interface import IWebScraper
from .services.web_scraper import DefaultWebScraper
from .services.web_scraper_factory import WebScraperFactory
from .services.embedding_service import EmbeddingService
from .services.article_repository import ArticleRepository
from .services.data_service import DataService
from .services.reranker_service import RerankerService


# ============== Web Scraper Factory ==============
@lru_cache()
def get_web_scraper_factory() -> WebScraperFactory:
    """Singleton web scraper factory instance."""
    return WebScraperFactory()


WebScraperFactoryDep = Annotated[WebScraperFactory, Depends(get_web_scraper_factory)]


# ============== Embedding Service ==============
@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Singleton embedding service."""
    return EmbeddingService()


EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]


# ============== Article Repository ==============
@lru_cache()
def get_article_repository() -> ArticleRepository:
    """Singleton article repository."""
    return ArticleRepository()


ArticleRepositoryDep = Annotated[ArticleRepository, Depends(get_article_repository)]


# ============== Reranker Service ==============
@lru_cache()
def get_reranker_service() -> RerankerService:
    """Singleton reranker service."""
    return RerankerService()


RerankerServiceDep = Annotated[RerankerService, Depends(get_reranker_service)]


# ============== Data Service ==========================
def get_data_service(
    scraper_factory: WebScraperFactoryDep,
    embedding_service: EmbeddingServiceDep,
    article_repository: ArticleRepositoryDep,
    reranker_service: RerankerServiceDep,
) -> DataService:
    """
    DataService with all dependencies injected.
    Not a singleton - gets fresh instance per request with injected deps.
    """
    return DataService(
        scraper_factory=scraper_factory,
        embedding_service=embedding_service,
        article_repository=article_repository,
        reranker_service=reranker_service,
    )


DataServiceDep = Annotated[DataService, Depends(get_data_service)]
