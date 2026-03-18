"""
Data module - handles article scraping, embeddings, and similarity search.
"""
from .routes import router as data_router
from .dependencies import get_web_scraper_factory, get_embedding_service, get_article_repository
from .services.data_service import DataService
from .services.web_scraper_factory import WebScraperFactory
from ...shared.event_bus import register as bus_register


def _register_event_handlers():
    """Register event bus handlers for inter-module communication."""
    # Create a DataService instance using the singleton sub-services
    data_service = DataService(
        scraper_factory=get_web_scraper_factory(),
        embedding_service=get_embedding_service(),
        article_repository=get_article_repository(),
    )

    bus_register(
        "search_similar_paragraphs",
        lambda **kwargs: data_service.search_similar_paragraphs(kwargs["request"]),
    )


def register_data_module(app):
    """Register all data module components with the app."""
    app.include_router(data_router, prefix="/api/data", tags=["data processing"])
    _register_event_handlers()