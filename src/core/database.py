"""
Database session management for SQLAlchemy.
Provides session factory and context managers for database operations.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator
from .config import get_settings

settings = get_settings()

# Create synchronous engine with connection pooling (for migrations)
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=10,        # Number of connections to maintain
    max_overflow=20,     # Maximum overflow connections
    echo=False,          # Set to True for SQL query logging
    connect_args={
        'options': '-c client_encoding=utf8'
    }
)

# Create async engine
async_database_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
async_engine = create_async_engine(
    async_database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Automatically handles commit/rollback and session cleanup.
    
    Usage:
        with get_db_session() as session:
            session.add(obj)
            # commit happens automatically on successful exit
            # rollback happens automatically on exception
    
    Yields:
        Session: SQLAlchemy database session
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI routes to get database session.
    Use with FastAPI's Depends() for automatic session management.
    
    Usage:
        @router.get("/")
        def route(db: Session = Depends(get_db)):
            ...
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.
    Automatically handles commit/rollback and session cleanup.
    
    Usage:
        async with get_async_db_session() as session:
            await session.add(obj)
            # commit happens automatically on successful exit
            # rollback happens automatically on exception
    
    Yields:
        AsyncSession: SQLAlchemy async database session
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async dependency for FastAPI routes to get database session.
    Use with FastAPI's Depends() for automatic session management.
    
    Usage:
        @router.get("/")
        async def route(db: AsyncSession = Depends(get_async_db)):
            ...
    
    Yields:
        AsyncSession: SQLAlchemy async database session
    """
    async with AsyncSessionLocal() as session:
        yield session
