from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .app_routes import base_router
from .modules.data import register_data_module
from .modules.data.dependencies import get_web_scraper_factory
from .modules.data.services.web_scraper import DefaultWebScraper
from .modules.rag import register_rag_module
from .modules.seo_generation import register_seo_generation_module
from .core.exceptions.bad_request_exception import BadRequestException
from .core.cache_service import cache_service
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import logging
import seqlog
import uuid
from .core.config import get_settings
from .core.security import limiter

# Get settings
settings = get_settings()

# Configure logging
log_level = logging.DEBUG if settings.is_development else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
      #  logging.FileHandler('app.log')  # File output
    ]
)

# Configure Seq logging if enabled
if settings.SEQ_ENABLED and settings.SEQ_SERVER_URL:
    seqlog.log_to_seq(
        server_url=settings.SEQ_SERVER_URL,
        api_key=settings.SEQ_API_KEY if settings.SEQ_API_KEY else None,
        level=logging.INFO,
        batch_size=10,
        auto_flush_timeout=2,
    )
    logging.info("Seq logging enabled")

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to handle correlation IDs for request tracing."""
    
    async def dispatch(self, request: Request, call_next):
        # Check if correlation ID exists in headers
        correlation_id = request.headers.get('X-Correlation-ID')
        
        # Only generate new correlation ID if one doesn't exist
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Store correlation ID in request state
        request.state.correlation_id = correlation_id
        
        # Log request with correlation ID
        logger.info(
            f"Incoming request: {request.method} {request.url.path} 'correlation_id' : {correlation_id}",
            extra={
                'correlation_id': correlation_id,
                'method': request.method,
                'path': request.url.path,
                'client_host': request.client.host if request.client else None
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers['X-Correlation-ID'] = correlation_id
        
        # Log response with correlation ID
        logger.info(
            f"Response: {response.status_code}",
            extra={
                'correlation_id': correlation_id,
                'status_code': response.status_code,
                'method': request.method,
                'path': request.url.path
            }
        )
        
        return response


# Create app instance
docs_url = "/docs" if not settings.is_production else None
redoc_url = "/redoc" if not settings.is_production else None
openapi_url = "/openapi.json" if not settings.is_production else None

app = FastAPI(
    title="NLP Semantic Lab",
    description="A hands-on learning laboratory for exploring NLP, semantic search, RAG, and embeddings",
    version="1.0.0",
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
)

# Rate limiter state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add correlation ID middleware
app.add_middleware(CorrelationIdMiddleware)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info(f"Starting application in [{settings.ENV}] environment...")

    # Auto-run Alembic migrations in development
    if settings.is_development:
        try:
            from alembic.config import Config as AlembicConfig
            from alembic import command as alembic_command

            alembic_cfg = AlembicConfig("alembic.ini")
            alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
            alembic_command.upgrade(alembic_cfg, "head")
            logger.info("Alembic migrations applied successfully")
        except Exception as e:
            logger.warning(f"Alembic migration failed: {e}")
    
    # ----- Register domain-specific web scrapers -----
    scraper_factory = get_web_scraper_factory()
    # The DefaultWebScraper is already set as the fallback.
    # Register it (or other IWebScraper implementations) for specific domains:
    scraper_factory.register("https://mxy.com", DefaultWebScraper())

    
    # scraper_factory.register("example.com", CustomScraper())
    logger.info("Web scraper factory configured")

    # Register all modules with app instance (includes router registration)
    register_data_module(app_instance)
    logger.info("Data module registered")
    
    register_rag_module(app_instance)
    logger.info("RAG module registered")
    
    register_seo_generation_module(app_instance)
    logger.info("SEO generation module registered")
    
    # Register base router
    app_instance.include_router(base_router)
    logger.info("Base router registered")
    
    # Initialize cache service
    await cache_service.connect()
    logger.info("Cache service connected")

    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await cache_service.disconnect()
    logger.info("Application shutdown complete")


# Set the lifespan
app.router.lifespan_context = lifespan



@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "service": "nlp-semantic-lab"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness check including dependencies."""
    redis_healthy = await cache_service.health_check()
    
    return {
        "status": "ready" if redis_healthy else "degraded",
        "checks": {
            "redis": redis_healthy
        }
    }


@app.exception_handler(BadRequestException)
async def bad_request_handler(request: Request, exc: BadRequestException):
    correlation_id = getattr(request.state, 'correlation_id', None)
    logger.warning(
        f"Bad request: {exc.response.error_description}",
        extra={
            'correlation_id': correlation_id,
            'status_code': exc.response.status_code,
            'path': request.url.path
        }
    )
    return JSONResponse(status_code=400, content={'status_code' :exc.response.status_code , 'error_description' :exc.response.error_description , 'data':None})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    correlation_id = getattr(request.state, 'correlation_id', None)
    logger.error(
        f"Unhandled exception: {str(exc)}",
        exc_info=True,
        extra={
            'correlation_id': correlation_id,
            'path': request.url.path,
            'method': request.method
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            'status_code': 9999,
            'error_description': 'An internal server error occurred.',
            'data': None
        }
    )

