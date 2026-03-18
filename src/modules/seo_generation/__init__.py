"""SEO Generation module."""
from .routes import router as seo_router


def register_seo_generation_module(app):
    """Register all SEO generation module components with the app."""
    app.include_router(seo_router, prefix="/api/seo", tags=["SEO Generation"])
