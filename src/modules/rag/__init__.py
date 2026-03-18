"""RAG (Retrieval-Augmented Generation) module."""
from .routes import router as rag_router


def register_rag_module(app):
    """Register all RAG module components with the app."""
    app.include_router(rag_router, prefix="/api/rag", tags=["RAG"])

